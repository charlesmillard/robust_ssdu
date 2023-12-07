from utils_n2n.utils import *
from torch.utils.data import Dataset
from model.mask_tools import *


class ZfData(Dataset):
    def __init__(self, categ='train', config=dict):
        self.__name__ = categ
        self.nx = config['data']['nx']
        self.ny = config['data']['ny']
        self.fixed_ncoil = config['data']['fixed_ncoil']
        self.sample_type = config['mask']['sample_type']
        self.dev = config['network']['device']
        self.sim_noise = config['noise']['sim_noise']
        self.data_norm = config['data']['norm']
        self.whiten_noise = config['noise']['whiten']
        self.whiten_sq_sz = config['noise']['whiten_sq_sz']

        self.prob_omega = gen_pdf(self.nx, self.ny, 1 / config['mask']['us_fac'], config['mask']['poly_order'],
                                  config['mask']['fully_samp_size'], self.sample_type)

        root = config['data']['loc'] + '/multicoil_'
        self.fileRoot = root + categ + '/'
        self.fileRootSmaps = root + categ + '_smaps/'
        self.file_list, self.slice_cumsum, self.len = self._select_slices(categ, config)

        print(categ + ' dataset contains {} slices'.format(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_idx = np.where(idx >= self.slice_cumsum)[0][-1]
        slice_idx = idx - self.slice_cumsum[file_idx]

        y0 = self._get_multicoil(file_idx, slice_idx)
        im_mask = self._get_mask(file_idx, slice_idx)

        set_seeds(idx)
        mask_omega = mask_from_prob(self.prob_omega, self.sample_type)
        noise1 = torch.randn(y0.shape).float()

        return (y0.to(self.dev), noise1.to(self.dev), mask_omega.to(self.dev),
                self.prob_omega.to(self.dev), im_mask.to(self.dev), \
               (self.file_list[file_idx], slice_idx.item()))

    def _get_mask(self, file_idx, slice_idx):
        file = self.file_list[file_idx][:-3]
        file_loc = self.fileRootSmaps + file + '_sl' + str(slice_idx) + '_mask.npy'
        if not torch.cuda.is_available() and self.__name__ == 'test':
            try:
                mask = np.load(file_loc)
                mask = torch.rot90(torch.as_tensor(mask))
                mask = mask.unsqueeze(0).unsqueeze(0)
            except:
                mask = 1
        else:
            mask = 1

        return torch.as_tensor(mask)

    def _get_multicoil(self, file_idx, slice_idx):
        file = self.file_list[file_idx]
        f = h5py.File(self.fileRoot + file, 'r')
        y0 = torch.as_tensor(f['kspace'][slice_idx])
        y0 = torch.flip(y0, [1])
        y0 = torch.permute(torch.view_as_real(y0), (3, 0, 1, 2))
        y0 = pad_or_trim_tensor(y0, self.nx, self.ny)

        if self.data_norm:
            y0 /= torch.max(kspace_to_rss(torch.unsqueeze(y0, 0)))

        if self.whiten_noise:
            # whiten and normalize to unit noise standard deviation
            # (useful for prospectively noisy data
            backg_mask = mask_corners(self.nx, self.ny, self.whiten_sq_sz)
            sig_inv = whitening_mtx(y0, backg_mask, True)
            y0 = whiten_kspace(y0, sig_inv)

        return y0.float()

    def _select_slices(self, categ, config):
        file_list = os.listdir(self.fileRoot)
        file_list_corrected = []
        nslices = []
        for file_idx in range(len(file_list)):
            file = file_list[file_idx]
            f = h5py.File(self.fileRoot + file, 'r')
            if f['kspace'].shape[1] == self.fixed_ncoil or self.fixed_ncoil is None:
                file_list_corrected.append(file)
                nslices.append(f['kspace'].shape[0])
        slice_cumsum = np.cumsum(nslices)
        slice_cumsum = np.insert(slice_cumsum, 0, 0)

        dataset_trunc = config['data']['train_trunc'] if categ == 'train' else config['data']['val_trunc']
        if dataset_trunc is not None:
            if len(file_list) > dataset_trunc:
                file_list_corrected = file_list_corrected[:dataset_trunc]
                nslices = nslices[:dataset_trunc]

        nslices_all = int(np.sum(nslices))

        return file_list_corrected, slice_cumsum, nslices_all
