import numpy as np
import torch
import random
from torch import nn, optim
import torchvision
import time
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def supervised_contrastive_loss(outputs, labels, temperature=0.1, num_classes=10, epsilon=0.0001):
    digit_indices = [np.where(labels.detach().numpy() == i)[0] for i in range(num_classes)]
    total_loss = 0.0
    for i in range(len(labels)):
        anchor_label = labels[i].item()
        positive_indices = digit_indices[anchor_label]
        negative_indices = list(digit_indices)
        _ = negative_indices.pop(anchor_label)
        negative_indices = np.concatenate(negative_indices)
        anchor_output = outputs[i]
        exp_inner_products = torch.exp(anchor_output @ outputs.T / temperature)

        # compute denominator
        neg_denom = 0.0
        for negative_index in negative_indices:
            neg_denom += exp_inner_products[negative_index]
        loss = 0.0
        for positive_index in positive_indices:
            if positive_index == i:
                continue
            loss -= torch.log(exp_inner_products[positive_index] / (neg_denom + exp_inner_products[positive_index]))
        loss /= len(positive_indices)
        total_loss += loss
    return total_loss

def supervised_contrastive_loss2(outputs, labels, augmented_outputs, temperature=0.1, num_classes=10):
    digit_indices = [np.where(labels.detach().numpy() == i)[0] for i in range(num_classes)]
    total_loss = 0.0
    for i in range(len(labels)):
        anchor_label = labels[i].item()
        positive_indices = digit_indices[anchor_label]
        negative_indices = list(digit_indices)
        _ = negative_indices.pop(anchor_label)
        negative_indices = np.concatenate(negative_indices)
        anchor_output = outputs[i]
        exp_inner_products = torch.exp(anchor_output @ outputs.T / temperature)
        exp_augmented_inner_products = torch.exp(anchor_output @ augmented_outputs.T / temperature)

        # compute denominator
        neg_denom = 0.0
        for negative_index in negative_indices:
            neg_denom += exp_inner_products[negative_index] + exp_augmented_inner_products[negative_index]
        loss = 0.0
        for positive_index in positive_indices:
            if positive_index == i:
                loss -= torch.log(exp_augmented_inner_products[positive_index] / (neg_denom + exp_augmented_inner_products[positive_index]))
                continue
            loss -= torch.log(exp_inner_products[positive_index] / (neg_denom + exp_inner_products[positive_index]))
            loss -= torch.log(exp_augmented_inner_products[positive_index] / (neg_denom + exp_augmented_inner_products[positive_index]))
        loss /= len(positive_indices)
        total_loss += loss
    return total_loss

def unsupervised_n_tuple_loss(outputs, labels, augmented_outputs, temperature=0.1):
    total_loss = 0.0
    for i in range(len(labels)):
        anchor_output = outputs[i]
        exp_inner_products = torch.exp(anchor_output @ outputs.T / temperature)
        exp_augmented_inner_products = torch.exp(anchor_output @ augmented_outputs.T / temperature)

        # compute denominator
        neg_denom = 0.0
        for j in range(len(labels)):
            if j == i:
                neg_denom += exp_augmented_inner_products[j]
                continue
            neg_denom += exp_inner_products[j] + exp_augmented_inner_products[j]
        total_loss -= torch.log(exp_augmented_inner_products[i] / neg_denom)
    return total_loss

def supervised_n_tuple_loss(outputs, labels, temperature=0.1, num_classes=10):
    digit_indices = [np.where(labels.detach().numpy() == i)[0] for i in range(num_classes)]
    total_loss = 0.0
    for i in range(len(labels)):
        anchor_label = labels[i].item()
        positive_indices = digit_indices[anchor_label]
        positive_index = random.choice(positive_indices)
        negative_indices = list(digit_indices)
        _ = negative_indices.pop(anchor_label)
        negative_indices = np.concatenate(negative_indices)
        anchor_output = outputs[i]
        exp_inner_products = torch.exp(anchor_output @ outputs.T / temperature)

        # compute denominator
        neg_denom = 0.0
        for negative_index in negative_indices:
            neg_denom += exp_inner_products[negative_index]
        total_loss -= torch.log(exp_inner_products[positive_index] / neg_denom)
    return total_loss

def supervised_triplet_loss(outputs, labels, margin=1.0, num_classes=10):
    digit_indices = [np.where(labels.detach().numpy() == i)[0] for i in range(num_classes)]
    criterion = nn.TripletMarginLoss(margin=margin)
    loss = 0.0
    for i in range(len(labels)):
        anchor_label = labels[i].item()
        positive_index = np.random.choice(digit_indices[anchor_label])
        negative_label = (anchor_label + random.randint(1, 9)) % num_classes
        if len(digit_indices[negative_label]) == 0:
            continue
        negative_index = np.random.choice(digit_indices[negative_label])
        anchor_output = outputs[i]
        positive_output = outputs[positive_index]
        negative_output = outputs[negative_index]
        loss += criterion(anchor_output, positive_output, negative_output)
    return loss

def unsupervised_triplet_loss(outputs, labels, augmented_outputs, margin=1.0, num_classes=10):
    criterion = nn.TripletMarginLoss(margin=margin)
    loss = 0.0
    batch_size = outputs.shape[0]
    for i in range(batch_size):
        anchor_output = outputs[i]
        positive_output = augmented_outputs[i]
        negative_output = outputs[(i + np.random.randint(batch_size)) % batch_size]
        loss += criterion(anchor_output, positive_output, negative_output)
    return loss

def augment_mnist(train_dataset):
    num_images = 60000
    augmented_training_data = train_dataset.data.detach().clone()
    for i in range(num_images):
        augmenter = torchvision.transforms.RandAugment()
        #augmenter = torch.nn.Sequential(torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.AugMix())
        augmented_image = augmenter.forward(augmented_training_data[i].unsqueeze(0))
        augmented_training_data[i] = augmented_image[0]
    augmented_training_data = (augmented_training_data.float() / 255.0).unsqueeze(1)
    return augmented_training_data.detach().numpy()

def augment_cifar(train_dataset):
    augmented_training_data = torch.tensor(train_dataset.data.transpose(0, 3, 1, 2))
    num_images = 50000
    batch_size = 100
    for i in range(num_images // batch_size):
        augmenter = torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10)
        #augmenter = torchvision.transforms.RandAugment()
        augmented_training_data[batch_size*i:batch_size*(i+1)] = augmenter.forward(augmented_training_data[batch_size*i:batch_size*(i+1)])
    augmented_training_data = augmented_training_data.float() / 255.0
    return augmented_training_data.detach().numpy()

def train_triplet(model, train_dataset, test_dataset, epochs=10, margin=1.0, optimizer=None, num_classes=10):
    train_data = train_dataset.data.float() / 255.0
    criterion = nn.TripletMarginLoss(margin=margin)

    # optimizer, I've used Adadelta, as it works well without any magic numbers
    if optimizer is None:
        optimizer = optim.Adadelta(model.parameters())

    start_ts = time.time()

    train_losses = []
    val_accuracies_regular = []
    val_accuracies_forward = []

    num_samples = train_dataset.data.shape[0]
    classes_range = np.arange(num_classes).astype(int)
    digit_indices = [np.where(train_dataset.targets.detach().numpy() == i)[0] for i in range(num_classes)]

    # loop for every epoch (training + evaluation)
    for epoch in range(epochs):
        total_loss = 0

        # progress bar (works in Jupyter notebook too!)
        progress = tqdm(enumerate(range(num_samples)), desc="Loss: ", total=num_samples)

        for i, _ in progress:
            # choose anchor and negative class
            classes = np.random.choice(classes_range, size=2, replace=False)
            anchor_class = classes[0]
            negative_class = classes[1]
            
            # choose anchor and positive index
            positive_indexes = np.random.choice(digit_indices[anchor_class], size=2, replace=False)
            anchor_index = positive_indexes[0]
            positive_index = positive_indexes[1]

            # choose negative index
            negative_index = random.choice(digit_indices[negative_class])

            # get corresponding inputs
            anchor_input = train_data[anchor_index]
            positive_input = train_data[positive_index]
            negative_input = train_data[negative_index]

            model.train()
            anchor_embedding = model(anchor_input.unsqueeze(0).unsqueeze(0))
            positive_embedding = model(positive_input.unsqueeze(0).unsqueeze(0))
            negative_embedding = model(negative_input.unsqueeze(0).unsqueeze(0))

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

            # updating progress bar
            progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
        
        # set model to evaluating (testing)
        model.eval()
        with torch.no_grad():
            embed_regular_train, embed_regular_test = embed_regular(model, train_dataset, test_dataset)
            svm_regular_embedding = LinearSVC().fit(embed_regular_train, train_dataset.targets)
            regular_score = svm_regular_embedding.score(embed_regular_test, test_dataset.targets)
            val_accuracies_regular.append(regular_score)

            embed_forward_train, embed_forward_test = embed_forward(model, train_dataset, test_dataset)
            svm_forward_embedding = LinearSVC().fit(embed_forward_train, train_dataset.targets)
            forward_score = svm_forward_embedding.score(embed_forward_test, test_dataset.targets)
            val_accuracies_forward.append(forward_score)
            
        print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss}, val accuracy regular: {val_accuracies_regular[-1]}, val accuracy forward: {val_accuracies_forward[-1]}")
        train_losses.append(total_loss / num_samples) # for plotting learning curve

    training_time = time.time()-start_ts
    print(f"Training time: {training_time}s")
    return train_losses, val_accuracies_regular, val_accuracies_forward, training_time

def augment_train(model, train_dataset, test_dataset, loss_function, epochs=10, batch_size=64, optimizer=None):
    #augmented_training_data = augment(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # optimizer, I've used Adadelta, as it works well without any magic numbers
    if optimizer is None:
        optimizer = optim.Adadelta(model.parameters())

    start_ts = time.time()

    train_losses = []
    batches = len(train_loader)
    val_accuracies_regular = []
    val_accuracies_forward = []
    val_ar_regular = []
    val_ar_forward = []

    # loop for every epoch (training + evaluation)
    for epoch in range(epochs):
        total_loss = 0

        # progress bar (works in Jupyter notebook too!)
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

        # ----------------- VALIDATION  ----------------- 
        # set model to evaluating (testing)
        model.eval()
        with torch.no_grad():
            # svm embedding
            embed_regular_train, embed_regular_test = embed_regular(model, train_dataset, test_dataset)
            svm_regular_embedding = LinearSVC().fit(embed_regular_train, train_dataset.targets)
            regular_score = svm_regular_embedding.score(embed_regular_test, test_dataset.targets)
            val_accuracies_regular.append(regular_score)

            # kmeans
            kmeans_regular_train = KMeans(n_clusters=10).fit(embed_regular_train)
            val_ar_regular.append(adjusted_rand_score(train_dataset.targets, kmeans_regular_train.predict(embed_regular_train)))

            # plot pca
            pca_embedding_regular = PCA(n_components=2).fit(embed_regular_train)
            pca_regular_projection = pca_embedding_regular.transform(embed_regular_test)
            plt.title('PCA Projection, Epoch ' + str(epoch + 1))
            plt.scatter(pca_regular_projection[:,0], pca_regular_projection[:,1], c=test_dataset.targets, cmap='tab10', s=1)
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.show()

            # svm forward
            embed_forward_train, embed_forward_test = embed_forward(model, train_dataset, test_dataset)
            svm_forward_embedding = LinearSVC().fit(embed_forward_train, train_dataset.targets)
            forward_score = svm_forward_embedding.score(embed_forward_test, test_dataset.targets)
            val_accuracies_forward.append(forward_score)

            # kmeans
            kmeans_forward_train = KMeans(n_clusters=10).fit(embed_forward_train)
            val_ar_forward.append(adjusted_rand_score(train_dataset.targets, kmeans_forward_train.predict(embed_forward_train)))
        print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, val accuracies regular: {val_accuracies_regular[-1]}, val accuracies forward: {val_accuracies_forward[-1]}")
        print(f"val accuracies regular: {val_ar_regular[-1]}, val accuracies forward: {val_ar_forward[-1]}")

        # ----------------- TRAINING  -------------------- 
        # set model to training
        model.train()
        
        for i, data in progress:
            X, y = data[0], data[1]

            augmenter1 = torchvision.transforms.RandAugment()
            augmenter2 = torchvision.transforms.RandAugment()
            augmented_X1 = torch.zeros(X.size())
            augmented_X2 = torch.zeros(X.size())
            for i in range(X.shape[0]):
                image_i = torchvision.transforms.ToPILImage()(X[i])

                augmented_image1 = augmenter1.forward(image_i)
                augmented_X1[i] = (torchvision.transforms.PILToTensor()(augmented_image1).float() / 255.0)

                augmented_image2 = augmenter2.forward(image_i)
                augmented_X2[i] = (torchvision.transforms.PILToTensor()(augmented_image2).float() / 255.0)
                
            
            # training step for single batch
            model.zero_grad()
            augmented_outputs1 = model(augmented_X1)
            augmented_outputs2 = model(augmented_X2)
            loss = loss_function(augmented_outputs1, y, augmented_outputs2)
            loss.backward()
            optimizer.step()

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

            # updating progress bar
            progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
        
    training_time = time.time()-start_ts
    print(f"Training time: {training_time}s")
    return train_losses, val_accuracies_regular, val_accuracies_forward, val_ar_regular, val_ar_forward, training_time

def train(model, train_dataset, test_dataset, loss_function, epochs=10, batch_size=64, optimizer=None):

    # params you need to specify:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # optimizer, I've used Adadelta, as it wokrs well without any magic numbers
    if optimizer is None:
        optimizer = optim.Adadelta(model.parameters())

    start_ts = time.time()

    train_losses = []
    val_accuracies_regular = []
    val_accuracies_forward = []
    batches = len(train_loader)
    

    # loop for every epoch (training + evaluation)
    for epoch in range(epochs):
        total_loss = 0

        # progress bar (works in Jupyter notebook too!)
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

        # ----------------- TRAINING  -------------------- 
        # set model to training
        model.train()
        
        for i, data in progress:
            X, y = data[0], data[1]
            
            # training step for single batch
            model.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

            # updating progress bar
            progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
        
        # ----------------- VALIDATION  -----------------
        
        # set model to evaluating (testing)
        model.eval()
        with torch.no_grad():
            embed_regular_train, embed_regular_test = embed_regular(model, train_dataset, test_dataset)
            svm_regular_embedding = LinearSVC().fit(embed_regular_train, train_dataset.targets)
            regular_score = svm_regular_embedding.score(embed_regular_test, test_dataset.targets)
            val_accuracies_regular.append(regular_score)

            pca_embedding_regular = PCA(n_components=2).fit(embed_regular_train)
            pca_regular_projection = pca_embedding_regular.transform(embed_regular_test)
            plt.title('PCA Projection, Epoch ' + epoch)
            plt.scatter(pca_regular_projection[:,0], pca_regular_projection[:,1], c=test_dataset.targets, cmap='tab10', s=1)
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.show()

            embed_forward_train, embed_forward_test = embed_forward(model, train_dataset, test_dataset)
            svm_forward_embedding = LinearSVC().fit(embed_forward_train, train_dataset.targets)
            forward_score = svm_forward_embedding.score(embed_forward_test, test_dataset.targets)
            val_accuracies_forward.append(forward_score)
        
        print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, val accuracies regular: {val_accuracies_regular[-1]}, val accuracies forward: {val_accuracies_forward[-1]}")
        train_losses.append(total_loss/batches) # for plotting learning curve
    training_time = time.time()-start_ts
    print(f"Training time: {training_time}s")
    return train_losses, val_accuracies_regular, val_accuracies_forward, training_time

def embed_regular(model, train_dataset, test_dataset):
    batch_size = 100
    train_data = train_dataset.data.float() / 255.0
    num_train = train_data.shape[0]
    embedding_dim = 160
    train_embedding = np.zeros((num_train, embedding_dim))
    for i in range(num_train // batch_size):
        batch_to_embed = train_data[batch_size*i:batch_size*(i+1)].reshape(batch_size, 1, 28, 28)
        embeddingi = model.embed(batch_to_embed).detach().numpy()
        train_embedding[batch_size*i:batch_size*(i+1)] = embeddingi

    test_data = test_dataset.data.float() / 255.0
    num_test = test_data.shape[0]
    test_embedding = np.zeros((num_test, embedding_dim))
    for i in range(num_test // batch_size):
        batch_to_embed = test_data[batch_size*i:batch_size*(i+1)].reshape(batch_size, 1, 28, 28)
        embeddingi = model.embed(batch_to_embed).detach().numpy()
        test_embedding[batch_size*i:batch_size*(i+1)] = embeddingi
    
    return train_embedding, test_embedding

def embed_forward(model, train_dataset, test_dataset):
    batch_size = 100
    train_data = train_dataset.data.float() / 255.0
    num_train = train_data.shape[0]
    embedding_dim = 80
    train_embedding = np.zeros((num_train, embedding_dim))
    for i in range(num_train // batch_size):
        batch_to_embed = train_data[batch_size*i:batch_size*(i+1)].reshape(batch_size, 1, 28, 28)
        embeddingi = model(batch_to_embed).detach().numpy()
        train_embedding[batch_size*i:batch_size*(i+1)] = embeddingi

    test_data = test_dataset.data.float() / 255.0
    num_test = test_data.shape[0]
    test_embedding = np.zeros((num_test, embedding_dim))
    for i in range(num_test // batch_size):
        batch_to_embed = test_data[batch_size*i:batch_size*(i+1)].reshape(batch_size, 1, 28, 28)
        embeddingi = model(batch_to_embed).detach().numpy()
        test_embedding[batch_size*i:batch_size*(i+1)] = embeddingi
    
    return train_embedding, test_embedding

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })