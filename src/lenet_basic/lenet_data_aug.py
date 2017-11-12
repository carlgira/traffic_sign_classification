import src.lenet_basic.lenet_basic as lb

if __name__ == '__main__':
	nn = lb.LeNETBasic('lenet_data_aug', data_aug=True)
	nn.train_nn()
