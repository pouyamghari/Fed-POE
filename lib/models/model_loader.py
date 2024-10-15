from keras.models import load_model

def model_loader(args):
    if args.dataset=="CIFAR-10":
        model = load_model('lib/models/vgg_2_0_cifar.h5')
        return model
    elif args.dataset=="FMNIST":
        model = load_model('lib/models/model_fmnist.h5')
        return model