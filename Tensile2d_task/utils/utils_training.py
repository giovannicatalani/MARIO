import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def training_step(
    func_rep,
    graph,
    latent_cond = True,
    return_reconstructions=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    func_rep.zero_grad()
    conditions = graph.cond
    coords = graph.input
    features = graph.output
    features_scalars = graph.output_scalars

    loss = 0

    with torch.set_grad_enabled(True):
        if latent_cond:
            features_recon = func_rep.modulated_forward(coords, conditions[graph.batch])
            scalar_pred = func_rep.predict_scalars(conditions)
        else:
            features_recon = func_rep(coords)

        field_loss = ((features_recon - features) ** 2).mean()
        scalar_loss = ((scalar_pred - features_scalars) ** 2).mean()
        loss = field_loss + 0.5*scalar_loss

    outputs = {"loss": loss, "field_loss": field_loss, "scalar_loss": scalar_loss}

    if return_reconstructions:
        outputs["reconstructions"] = features_recon
        
    return outputs


def graph_inner_loop(
    func_rep,
    modulations,
    coords,
    features,
    batch_index,
    inner_steps,
    inner_lr,
    is_train=False,
):
    """Performs inner loop, i.e. fits modulations such that the function
    representation can match the target features.


    """
    fitted_modulations = modulations
    for step in range(inner_steps):
        fitted_modulations = graph_inner_loop_step(
            func_rep,
            fitted_modulations,
            coords,
            features,
            batch_index,
            inner_lr,
            is_train,
        )

    return fitted_modulations


def graph_inner_loop_step(
    func_rep,
    modulations,
    coords,
    features,
    batch_index,
    inner_lr,
    is_train=False,
):
    """Performs a single inner loop step."""
    detach = False
    batch_size = modulations.shape[0]
    loss = 0
    with torch.enable_grad():
        # Note we multiply by batch size here to undo the averaging across batch
        # elements from the MSE function. Indeed, each set of modulations is fit
        # independently and the size of the gradient should not depend on how
        # many elements are in the batch

        features_recon = func_rep.modulated_forward(coords, modulations[batch_index])
        loss = ((features_recon - features) ** 2).mean() * batch_size
        
        # If we are training, we should create graph since we will need this to
        # compute second order gradients in the MAML outer loop
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train and not detach,
        )[0]

    # Perform single gradient descent step
    return modulations - inner_lr * grad


def graph_outer_step(
    func_rep,
    graph,
    inner_steps,
    inner_lr,
    is_train=False,
    detach_modulations=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    func_rep.zero_grad()
    batch_size = len(graph)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = torch.zeros_like(graph.modulations).requires_grad_()
    coords = graph.input
    features = graph.output

    # Run inner loop
    modulations = graph_inner_loop(
        func_rep,
        modulations,
        coords,
        features,
        graph.batch,
        inner_steps,
        inner_lr,
        is_train,
    )

    if detach_modulations:
        modulations = modulations.detach()  # 1er ordre

    loss = 0
    batch_size = modulations.shape[0]

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords, modulations[graph.batch])
        loss = ((features_recon - features) ** 2).mean()

    outputs = {
        "loss": loss,
        "modulations": modulations,
    }

    return outputs