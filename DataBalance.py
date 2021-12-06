import numpy as np
import math
import random
import collections
from imgaug import augmenters as iaa
from phe import paillier
import datetime

from utils.enc import convert_ciphertext


class DataBalance:
    def __init__(self, dp, mediator_users_num):
        self.dp = dp  # a variable of class DataProcessor
        self.td = 0.1
        self.ta = -0.1
        self.mediator = []
        self.gamma = mediator_users_num  # the maximum number for a mediator can communicate

        self.mediator_distribution = dict()
        self.client_cipher_pool = dict()

        self.pk0, self.sk0 = paillier.generate_paillier_keypair()
        self.pk1, self.sk1 = paillier.generate_paillier_keypair()
        self.client_key_pairs = dict()
        for client in range(self.dp.size_device):
            key_pair = dict()
            key_pair['pk'], key_pair['sk'] = paillier.generate_paillier_keypair()
            self.client_key_pairs[client] = key_pair

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

        while client_pool:
            new_mediator = set()
            mediator_label_pool = np.array([])
            c2 = collections.Counter(self.dp.global_train_label)
            for key in c2.keys():
                self.mediator_distribution[key] = self.pk1.encrypt(0)
            while client_pool and len(new_mediator) < self.gamma:
                select_client, kl_score = None, float('inf')
                for client in client_pool:
                    r = random.randint(1, 1024)
                    d_cipher = dict()
                    for key in self.mediator_distribution.keys():
                        d_cipher[key] = (self.mediator_distribution[key] + self.client_cipher_pool[client][key]) * r
                        d_cipher[key] = self.sk1.decrypt(d_cipher[key])
                    new_kl_score = self.dp.get_kl_divergence_enc(self.dp.global_train_label, d_cipher)
                    if new_kl_score < kl_score:
                        select_client = client
                new_mediator.add(select_client)
                mediator_label_pool = np.hstack([mediator_label_pool, self.dp.local_train_label[select_client]])
                client_pool.remove(select_client)
                for key in self.mediator_distribution.keys():
                    self.mediator_distribution[key] += self.client_cipher_pool[select_client][key]
            self.mediator.append(new_mediator)

    def assign_clients_col(self, balance=True):
        # assign the devices to each mediator using greedy algorithm
        if not balance:
            self.mediator = [{i} for i in range(self.dp.size_device)]
            return

        client_pool = set([i for i in range(self.dp.size_device)])

        # 用客户端的公钥加密客户端数据
        for client in client_pool:
            c1 = collections.Counter(self.dp.local_train_label[client])
            cipher = dict()
            for key in collections.Counter(self.dp.global_train_label).keys():
                if key in c1.keys():
                    cipher[key] = self.client_key_pairs[client]['pk'].encrypt(c1[key])
                else:
                    cipher[key] = self.client_key_pairs[client]['pk'].encrypt(0)
            self.client_cipher_pool[client] = cipher

        while client_pool:
            new_mediator = set()
            mediator_label_pool = np.array([])
            mediator_distribution_cipher = dict()
            c2 = collections.Counter(self.dp.global_train_label)
            # 用各个客户端的密钥加密当前协调者的分布
            for client in client_pool:
                mediator_distribution = dict()
                for key in c2.keys():
                    mediator_distribution[key] = self.client_key_pairs[client]['pk'].encrypt(0)
                mediator_distribution_cipher[client] = mediator_distribution

            while client_pool and len(new_mediator) < self.gamma:
                select_client, kl_score = None, float('inf')
                for client in client_pool:
                    r = random.randint(1, 1024)
                    d_cipher = dict()
                    for key in mediator_distribution_cipher[client].keys():
                        d_cipher[key] = (mediator_distribution_cipher[client][key] + self.client_cipher_pool[client][key]) * r
                        d_cipher[key] = self.client_key_pairs[client]['sk'].decrypt(d_cipher[key])
                    new_kl_score = self.dp.get_kl_divergence_enc(self.dp.global_train_label, d_cipher)
                    if new_kl_score < kl_score:
                        select_client = client
                new_mediator.add(select_client)
                mediator_label_pool = np.hstack([mediator_label_pool, self.dp.local_train_label[select_client]])
                client_pool.remove(select_client)
                # 将新的协调者分布密文转换成用各个客户端公钥加密的密文
                for client in client_pool:
                    for key in c2.keys():
                        cipher = convert_ciphertext(self.client_key_pairs[select_client]['sk'],
                                                    self.client_key_pairs[select_client]['pk'],
                                                    self.client_key_pairs[client]['pk'],
                                                    self.client_cipher_pool[select_client][key])
                        mediator_distribution_cipher[client][key] += cipher

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

        starttime = datetime.datetime.now()

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

        endtime = datetime.datetime.now()
        print("merging time: {m}".format(m=(endtime - starttime).microseconds))

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

    def z_score_enc(self):
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

        starttime = datetime.datetime.now()

        # 2 : Initialize
        r_ad = np.zeros(self.dp.size_class)

        # 3 : Calculate the data size of each class C
        num_each_class_cipher = dict()
        for key in collections.Counter(self.dp.global_train_label).keys():
            num_each_class_cipher[key] = self.pk1.encrypt(0)
        client_pool = set([i for i in range(self.dp.size_device)])
        for client in client_pool:
            c1 = collections.Counter(self.dp.local_train_label[client])
            cipher = dict()
            for key in c1.keys():
                cipher[key] = self.pk1.encrypt(c1[key])
                num_each_class_cipher[key] += cipher[key]
            self.client_cipher_pool[client] = cipher

        num_each_class = np.zeros(self.dp.size_class)
        for key in num_each_class_cipher.keys():
            num_each_class[key] = self.sk1.decrypt(num_each_class_cipher[key])

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

        endtime = datetime.datetime.now()
        print("merging time: {m}".format(m=(endtime - starttime).microseconds))

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

    def z_score_col(self):
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

        starttime = datetime.datetime.now()

        # 2 : Initialize
        r_ad = np.zeros(self.dp.size_class)

        # 3 : Calculate the data size of each class C
        num_each_class = np.zeros(self.dp.size_class)
        num_each_class_cipher = dict()
        client_pool = set([i for i in range(self.dp.size_device)])
        random_pool = dict()

        # 算法1 - 合并各个客户端的分布
        for client in client_pool:
            c1 = collections.Counter(self.dp.local_train_label[client])
            r_i = random.randint(1, 1024)
            random_pool[client] = self.client_key_pairs[client]['pk'].encrypt(r_i)
            for key in collections.Counter(self.dp.global_train_label).keys():
                if key in c1:
                    num_each_class[key] += c1[key] + r_i
                else:
                    num_each_class[key] += r_i

        for key in collections.Counter(self.dp.global_train_label).keys():
            num_each_class_cipher[key] = self.client_key_pairs[0]['pk'].encrypt(num_each_class[key])

        for client in range(self.dp.size_device - 1):
            for key in num_each_class_cipher.keys():
                # 算法1第9行
                num_each_class_cipher[key] -= random_pool[client]
                # 算法1第10行
                num_each_class_cipher[key] = convert_ciphertext(self.client_key_pairs[client]['sk'],
                                                                self.client_key_pairs[client]['pk'],
                                                                self.client_key_pairs[client+1]['pk'],
                                                                num_each_class_cipher[key])
        for key in num_each_class_cipher.keys():
            # 算法1第11行
            num_each_class_cipher[key] -= random_pool[self.dp.size_device-1]
            # 算法1第12行
            num_each_class_cipher[key] = convert_ciphertext(self.client_key_pairs[self.dp.size_device-1]['sk'],
                                                            self.client_key_pairs[self.dp.size_device-1]['pk'],
                                                            self.pk0,
                                                            num_each_class_cipher[key])
            # 算法1第13行
            num_each_class[key] = self.sk0.decrypt(num_each_class_cipher[key])

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

        endtime = datetime.datetime.now()
        print("merging time: {m}".format(m=(endtime - starttime).microseconds))

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
