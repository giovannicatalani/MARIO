import airfrans as af
import pyoche as pch
from utils.funcs import process_dataset


train_dataset = af.dataset.load(root='/path/to/airfrans_dataset/', task='scarce', train=True)
test_dataset = af.dataset.load(root='/path/to/airfrans_dataset/', task='scarce', train=False)

pyoche_data = pch.MlDataset(process_dataset(train_dataset))
pyoche_data_test = pch.MlDataset(process_dataset(test_dataset))
pyoche_data.save_h5file('af_train.pch')
pyoche_data_test.save_h5file('af_test.pch')
