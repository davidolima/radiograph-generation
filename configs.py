class Configs:
    def __init__(self, model=None) -> None:
        self.root_dir = "/media/david/bb5b899e-a7e9-49dd-b267-5014035bf700/datasets/pan-radiographs/1st-set"
        self.csv = self.root_dir + "/pan-radiographs.csv"
        
        self.checkpoint_save_file_name = "checkpoint"

        if model:
            self.model = model.lower()

        self.n_epochs = 200
        self.batch_size = 64
        self.lr = 0.00005
        self.n_cpu = 8
        self.shuffle = True
        self.num_workers = 0
        self.mean = [0.485, 0.456, 0.406]
        self.stdv = [0.229, 0.224, 0.225]
        
        if model is None:
            self.latent_dim = 100
            self.img_size = 224
            self.channels = 3
            self.n_critic = 5
            self.clip_value = 0.01
            self.sample_interval = 400

        elif self.model == 'wgan-gp':
            self.checkpoint_save_file_name += ' ' + self.model
            self.latent_dim = 100
            self.img_size = 224
            self.channels = 3
            self.n_critic = 5
            self.sample_interval = 400
            self.lambda_gp = 10
