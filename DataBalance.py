import numpy as np
import math
import random
from imgaug import augmenters as iaa
from phe import paillier


class DataBalance:
    def __init__(self, dp):
        self.dp = dp  # a variable of class DataProcessor
        self.td = 0.1
        self.ta = -0.1
        self.mediator = []
        self.gamma = 3  # the maximum number for a mediator can communicate
        self.pk0, self.sk0 = paillier.generate_paillier_keypair()
        self.pk1, self.sk1 = paillier.generate_paillier_keypair()

    def assign_clients(self, balance=True):
        # assign the devices to each mediator using greedy algorithm
        if not balance:
            self.mediator = [{i} for i in range(self.dp.size_device)]
            return
        client_pool = set([i for i in range(self.dp.size_device)])
        while client_pool:
            new_mediator = set()
            mediator_label_pool = np.array([])
            while client_pool and len(new_mediator) < self.gamma:
                select_client, kl_score = None, float('inf')
                for client in client_pool:
                    new_kl_score = self.dp.get_kl_divergence(self.dp.global_train_label,
                                                             np.hstack([mediator_label_pool,
                                                                        self.dp.local_train_label[client]]))
                    if new_kl_score < kl_score:
                        select_client = client
                new_mediator.add(select_client)
                mediator_label_pool = np.hstack([mediator_label_pool, self.dp.local_train_label[select_client]])
                client_pool.remove(select_client)
            self.mediator.append(new_mediator)

    def assign_clients_enc(self, balance=True):
        # assign the devices to each mediator using greedy algorithm
        if not balance:
            self.mediator = [{i} for i in range(self.dp.size_device)]
            return
        client_pool = set([i for i in range(self.dp.size_device)])
        local_train_label_ciphertexts = np.array([])
        for client in client_pool:
            local_train_label_ciphertexts[client] = [self.pk1.encrypt(x) for x in self.dp.local_train_label[client]]
        while client_pool:
            new_mediator = set()
            mediator_label_pool = np.array([])
            mediator_label_pool = [self.pk1.encrypt(x) for x in mediator_label_pool]
            while client_pool and len(new_mediator) < self.gamma:
                select_client, kl_score = None, float('inf')
                for client in client_pool:
                    summ = np.hstack([mediator_label_pool, local_train_label_ciphertexts[client]])
                    r = random.randint(1, 2048)
                    summ = [r * x for x in summ]
                    summ = [self.sk1.decrypt(x) for x in summ]
                    new_kl_score = self.dp.get_kl_divergence_enc(self.dp.global_train_label, summ)
                    if new_kl_score < kl_score:
                        select_client = client
                new_mediator.add(select_client)
                mediator_label_pool = np.hstack([mediator_label_pool, self.dp.local_train_label[select_client]])
                client_pool.remove(select_client)
            self.mediator.append(new_mediator)

    def z_score(self):
        """
        The FL Server part (Algorithm 2)
        The td and ta are the downsampling threshold and augmentation threshold
        Set Ta = -1/Td , the recommended value of td is 3.0 or 3.5
        The Rad is the ratio we use to control how many augmentations are generated or how many samples are retained.
        N : Number of classes
        K : Total number of clients
        labels : All the data label
        Ydown : set of majority class
        Yaug : set of minority class
        datasets : K clients datasets
        """
        # 2 : Initialize
        r_ad = np.zeros(self.dp.size_class)

        # 3 : Calculate the data size of each class C
        num_each_class = np.zeros(self.dp.size_class)
        for i in self.dp.global_train_label:
            num_each_class[i] = num_each_class[i] + 1

        # 4 : Calculate the mean m and the standard deviation s of C
        mean = np.mean(num_each_class)
        std = np.std(num_each_class, ddof=1)
        if std == 0:
            return
        # 5 : Calculate the z-score
        z = (num_each_class- mean) / std
        # 6-12 :
        y_down = set()
        y_aug = set()
        for y in range(self.dp.size_class):
            if z[y] < self.ta:
                y_aug.add(y)
                r_ad[y] = (-std * math.sqrt(z[y] * self.ta) + mean) / num_each_class[y]
            elif z[y] > self.td:
                y_down.add(y)
                r_ad[y] = (std * math.sqrt(z[y] * self.td) + mean) / num_each_class[y]

        # 13 : Send Yaug, Ydown, Rad to all clients ===================================================
        """
        The Clients part (Algorithm 2)
        """
        # 15-22 :
        for k in range(self.dp.size_device):
            print('size: {}'.format(k))
            new_feature_array = np.empty([0, self.dp.size_feature])
            new_label = []
            for i in range(len(self.dp.local_train_feature[k])):
                if i % 1000 == 0:
                    print('the {}th feature'.format(i))
                x, y = self.dp.local_train_feature[k][i], self.dp.local_train_label[k][i]
                new_x, new_y = x, y
                if y in y_down:
                    new_x, new_y = self.down_sample(x, y, r_ad[y])
                elif y in y_aug:
                    aug_x, aug_y = self.augment(x, y, r_ad[y]-1)
                    if aug_x is not None:
                        new_feature_array = np.vstack([new_feature_array, aug_x])
                        new_label.append(aug_y)
                if new_x is not None:
                    new_feature_array = np.vstack([new_feature_array, new_x])
                    new_label.append(new_y)
            self.dp.local_train_feature[k] = new_feature_array
            self.dp.local_train_label[k] = np.array(new_label)
        self.dp.refresh_global_data()

    @staticmethod
    def down_sample(x, y, r_ad):
        if random.random() < r_ad:
            return x, y
        else:
            return None, None

    def augment(self, x, y, r_ad):
        if random.random() > r_ad:
            return None, None
        else:
            image = None
            if self.dp.data_source == 'cifar':
                image = x.reshape(32, 32, 3)
            elif self.dp.data_source == 'mnist':
                image = x.reshape(28, 28)

            rand_select = random.random()
            image_aug = None
            image = image.astype(np.uint8)
            # augment the new image
            if rand_select < 0.25:
                image_aug = self.rotate(image)
            elif rand_select < 0.5:
                image_aug = self.shear(image)
            elif rand_select < 0.75:
                image_aug = self.scale(image)
            elif rand_select < 1:
                image_aug = self.shift(image)
            return image_aug.reshape(-1), y

    @staticmethod
    def rotate(image):
        # rotate image randomly in -25 - 25 degree
        rotate = iaa.Affine(rotate=(-25, 25))
        image_aug = rotate(image=image)
        return image_aug

    @staticmethod
    def shear(image):
        # shear image randomly between -25 and 25 degree
        aug_x = iaa.ShearX((-25, 25))
        aug_y = iaa.ShearY((-25, 25))
        image_aug = aug_x(image=image)
        image_aug = aug_y(image=image_aug)
        return image_aug

    @staticmethod
    def scale(image):
        # scale image randomly between 0.5 and 1.5
        aug1 = iaa.ScaleX((0.5, 1.5))
        aug2 = iaa.ScaleY((0.5, 1.5))
        image_aug = aug1(image=image)
        image_aug = aug2(image=image_aug)
        return image_aug

    @staticmethod
    def shift(image):
        # shift image randomly 10 percent
        aug1 = iaa.TranslateX(percent=(-0.1, 0.1))
        aug2 = iaa.TranslateY(percent=(-0.1, 0.1))
        image_aug = aug1(image=image)
        image_aug = aug2(image=image_aug)
        return image_aug


if __name__ == '__main__':
    print('self balance functions')
