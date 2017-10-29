import src.lenet_basic.lenet_basic as lb

if __name__ == '__main__':
	nn = lb.LeNETBasic('lenet', data_aug=False)
	nn.train_nn()