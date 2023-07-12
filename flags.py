import argparse
import os
import tensorflow as tf
import pickle


class Flags():
  def __init__(self):
    self.initialized = False
  
  def initialize(self, parser):
    # experiment specifics
    parser.add_argument('--name', type=str, default='pcxgan_flags', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--exp_name', type=str, default='pcxgan_flags', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--data_path', type=str, default='/media/aisec-102/DATA31/rachel/PCGAN/data/folds1234.npz', help='Data .npz file' )# CT_MRI-512-Updated
    parser.add_argument('--test_fold', type=int, default=5, help='fold to be left out of trainings' )
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--sample_dir', type=str, default='./training_samples', help='samples are saved here')
    parser.add_argument('--result_logs', type=str, default='./results' , help='Logs are stored here' )
    parser.add_argument('--model_path', type=str, default='./models/' , help='Model is stored here' )
    parser.add_argument('--hist_path', type=str, default='./history/' , help='Model is stored here' )
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    
    # input/output sizes
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--load_size', type=int, default=256, help='Scale images to this size. The final image will be cropped to --crop_size.')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
    parser.add_argument('--edge_threshold', type=float, default=0.03, help='Threshold for binarizing edge information.')
    parser.add_argument('--data_test_rate', type=float, default=0.2 , help='Test data proportion from the input data file.' )
    parser.add_argument('--apply_normalization', type=bool, default=False , help='Apply normalization.' )
    parser.add_argument('--remove_bad_images', type=bool, default=False , help='Remove bad images.' )
    
    # model specifics
    parser.add_argument('--latent_dim', type=int, default=256 , help='' )
    parser.add_argument('--feature_loss_coeff', type=float, default=4.0 , help='' )
    parser.add_argument('--vgg_feature_loss_coeff', type=float, default=10 , help='' )
    parser.add_argument('--kl_divergence_loss_coeff', type=float, default=0.05 , help='' )
    parser.add_argument('--generator_loss_coeff', type=float, default=0.05 , help='' )
    parser.add_argument('--identity_loss_coeff', type=float, default=5 , help='' )
    parser.add_argument('--cycle_loss_coeff', type=float, default=10 , help='' )
    parser.add_argument('--ssim_loss_coeff', type=float, default=2, help='' )
    parser.add_argument('--mae_loss_coeff', type=float, default=10.0 , help='' )
    parser.add_argument('--disc_loss_coeff', type=float, default=0.5, help='' )
    parser.add_argument('--gen_lr', type=float, default=2e-4 , help='' )
    parser.add_argument('--disc_lr', type=float, default=2e-4 , help='' )
    parser.add_argument('--gen_beta_1', type=float, default=0.0 , help='' )
    parser.add_argument('--gen_beta_2', type=float, default=0.9 , help='' )
    parser.add_argument('--disc_beta_1', type=float, default=0.5, help='' )
    parser.add_argument('--disc_beta_2', type=float, default=0.999 , help='' )
    parser.add_argument('--loss_weights', type=list, default=[1,100] , help='' )
    parser.add_argument('--result_log', type=str, default='results.log' , help='' )
    
    # encoder/decoder options
    parser.add_argument('--s_epsilon', type=float, default= 1e-5, help='' )
    parser.add_argument('--s_gamma_filters', type=int, default=128 , help='' )
    parser.add_argument('--s_gamma_filter_size', type=int, default=3 , help='' )
    parser.add_argument('--s_beta_filters', type=int, default=128, help='' )
    parser.add_argument('--s_beta_filter_size', type=int, default=3 , help='' )
    parser.add_argument('--e_n_filters', type=int, default=64 , help='' )
    parser.add_argument('--e_filter_size', type=int, default=3 , help='' )
    parser.add_argument('--strides', type=int, default=2 , help='' )
    parser.add_argument('--d_res_filters', type=int, default=1024 , help='' )
    parser.add_argument('--d_n_filters', type=int, default=256 , help='' )
    parser.add_argument('--d_filter_size', type=int, default=4 , help='' )
    parser.add_argument('--disc_n_filters', type=int, default=64 , help='' )
    parser.add_argument('--disc_filter_size', type=int, default=4 , help='' )
    parser.add_argument('--disc_strides', type=int, default=2 , help='' )
    
    
    
    # Monitor options
    parser.add_argument('--epochs', type=int, default=500, help='')
    parser.add_argument('--epoch_interval', type=int, default=10 , help='' )
    parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
    
    
    self.initialized = True
    return parser
  
  def gather_options(self):
    # initialize parser with basic options
    if not self.initialized:
      parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
      parser = self.initialize(parser)
    
    # get the basic options
    opt, unknown = parser.parse_known_args()
    if opt.load_from_opt_file:
      parser = self.update_options_from_file(parser, opt)
    
    opt = parser.parse_args()
    opt.name = opt.exp_name
    self.parser = parser
    return opt
  
  def print_options(self, opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
      comment = ''
      default = self.parser.get_default(k)
      if v != default:
        comment = '\t[default: %s]' % str(default)
      message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
  
  def option_file_path(self, opt, makedir=False):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if makedir:
      self.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt')
    return file_name
  
  def mkdirs(self, paths):
    if isinstance(paths, list) and not isinstance(paths, str):
      for path in paths:
        self.mkdir(path)
    else:
      self.mkdir(paths)
  
  def mkdir(self, path):
    if not os.path.exists(path):
      os.makedirs(path)
  
  def save_options(self, opt):
    file_name = self.option_file_path(opt, makedir=True)
    with open(file_name + '.txt', 'wt') as opt_file:
      for k, v in sorted(vars(opt).items()):
        comment = ''
        default = self.parser.get_default(k)
        if v != default:
          comment = '\t[default: %s]' % str(default)
        opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))
    
    with open(file_name + '.pkl', 'wb') as opt_file:
      pickle.dump(opt, opt_file)
  
  def update_options_from_file(self, parser, opt):
    new_opt = self.load_options(opt)
    for k, v in sorted(vars(opt).items()):
      if hasattr(new_opt, k) and v != getattr(new_opt, k):
        new_val = getattr(new_opt, k)
        parser.set_defaults(**{k: new_val})
    return parser
  
  def load_options(self, opt):
    file_name = self.option_file_path(opt, makedir=False)
    new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
    return new_opt
  
  def parse(self, save=False):
    
    opt = self.gather_options()
    opt.isTrain = True#self.isTrain   # train or test
    
    self.print_options(opt)
    if opt.isTrain:
      self.save_options(opt)
    
    # set gpu ids
    """
    
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
      id = int(str_id)
      if id >= 0:
        opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
      gpus = tf.config.experimental.list_physical_devices('GPU')
      if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
          tf.config.experimental.set_visible_devices(gpus[opt.gpu_ids[0]], 'GPU')
        except RuntimeError as e:
          # Visible devices must be set at program startup
          print(e)
    
    assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
      "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
      % (opt.batch_size, len(opt.gpu_ids))
    """
    self.opt = opt
    return self.opt
