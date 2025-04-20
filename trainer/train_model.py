import os.path
import torch
import numpy as np
import pandas as pd
from utils import metrics
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

def train_one_epoch(args, model, train_loader, loss_func, optimizer,threshold):
    model.train()

    total_train_loss = 0
    sample_num = 0
    all_trues = []
    all_scores = []

    # train
    for batch_idx, data in enumerate(tqdm(train_loader)):
        X, y_pos, mask_all, gene_name = data  # X:torch.Size([128, 3, 224, 224]),y_pos:torch.Size([128]),mask_all:torch.Size([128])
        sample_num += y_pos.size(0)
        # X = X.unsqueeze(0)          # torch.Size([1, 128, 224, 224]) I,B,W,H(in_channal, batch_size, weight, height) dtype:float32
        # X = X.permute(1, 0, 2, 3)   # torch_size([128, 1, 224, 224]) B,I,W,H

        y = y_pos.to(torch.float32)
        y = y.reshape(y.shape[0], 1)  # torch.Size([32,])--->torch.Size([32,1])
        if torch.cuda.is_available():
            X = X.to(args.device)
            y = y.to(args.device)

        y_pred = model(X)                # logits without sigmoid
        loss = loss_func(y_pred, y)
        total_train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_trues.append(y.data.cpu().numpy())
        sigmoid_y_pred = torch.sigmoid(torch.tensor(y_pred))
        all_scores.append(sigmoid_y_pred.data.cpu().numpy())

    all_trues_one_epoch = np.concatenate(all_trues, axis=0)
    all_scores_one_epoch = np.concatenate(all_scores, axis=0)
    acc, pre, rec, f1, AUC, AUPR = metrics.compute_metrics(all_trues_one_epoch, all_scores_one_epoch, threshold)

    return all_trues_one_epoch, all_scores_one_epoch, total_train_loss / sample_num, total_train_loss, acc, pre, rec, f1, AUC, AUPR

@torch.no_grad()
def val_one_epoch(args, model, val_loader, loss_func, threshold, test_mode=False):
    model.eval()

    total_val_loss = 0
    all_trues = []
    all_scores = []
    gene_names =[]
    sample_num = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader)):
            X, y_pos, mask_all, gene_name = data
            sample_num += y_pos.size(0)

            # X = X.unsqueeze(0)
            # X = X.permute(1, 0, 2, 3)

            y = y_pos.to(torch.float32)
            y = y.reshape(y.shape[0], 1)

            if torch.cuda.is_available():
                X = X.to(args.device)
                y = y.to(args.device)

            y_pred = model(X)
            loss = loss_func(y_pred, y)
            total_val_loss += loss.item()

            all_trues.append(y.data.cpu().numpy())
            sigmoid_y_pred = torch.sigmoid(torch.tensor(y_pred))
            all_scores.append(sigmoid_y_pred.data.cpu().numpy())
            gene_names.append(gene_name)

        all_trues_one_epoch = np.concatenate(all_trues, axis=0)
        all_scores_one_epoch = np.concatenate(all_scores, axis=0)
        all_gene_names_one_epoch = np.concatenate(gene_names, axis=0)
        acc, pre, rec, f1, AUC, AUPR = metrics.compute_metrics(all_trues_one_epoch, all_scores_one_epoch, threshold)
        if test_mode:
            return all_gene_names_one_epoch, all_trues_one_epoch, all_scores_one_epoch, total_val_loss / sample_num, total_val_loss, acc, pre, rec, f1, AUC, AUPR
        else:
            return all_trues_one_epoch, all_scores_one_epoch, total_val_loss / sample_num, total_val_loss, acc, pre, rec, f1, AUC, AUPR



def train(args, train_config, train_set, test_loader, model, loss_func, optimizer, threshold):
    print('\n---------------Start training!---------------')
    print("Device: ", args.device)
    print("Seed: ", args.seed)

    # On the training set: extract the data corresponding to train_set from dataset_mydata
    train_set_data = [train_set.dataset.dataset[int(np.array(train_set.dataset.indices[idx], np.int32))][0] for idx in train_set.indices]
    train_set_y_pos = [train_set.dataset.dataset[int(np.array(train_set.dataset.indices[idx], np.int32))][1] for idx in train_set.indices]
    train_set_mask_all = [train_set.dataset.dataset[int(np.array(train_set.dataset.indices[idx], np.int32))][2] for idx in train_set.indices]
    train_set_gene_name_all = [train_set.dataset.dataset[int(np.array(train_set.dataset.indices[idx], np.int32))][3] for idx in train_set.indices]

    # Packaging for data pairs
    sampled_train_set = [(X, y, mask, gn) for X, y, mask, gn in zip(train_set_data, train_set_y_pos, train_set_mask_all, train_set_gene_name_all)]

    # k-fold cross validation
    kfold_val_auprs = []
    kfold_val_aucs = []
    kfold_test_trues, kfold_test_scores, kfold_test_aucs, kfold_test_auprs = [], [], [], []

    patience = train_config['patience']
    kfold = train_config['kfold']

    skf = StratifiedKFold(n_splits=kfold, random_state=args.seed, shuffle=True)
    # Split all the training data into K folds, where K-1 folds are the new training set and 1 fold is the validation set
    for i, (train_index, val_index) in enumerate(skf.split(train_set_data, train_set_y_pos)):
        print(f'\nStart training CV fold {i + 1}:')
        train_sampler, val_sampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
        train_loader = DataLoader(sampled_train_set, batch_size=train_config['batch_size'], sampler=train_sampler)
        val_loader = DataLoader(sampled_train_set, batch_size=train_config['batch_size'], sampler=val_sampler)

        # Train model
        count = 0
        best_val_aupr = .0
        best_val_auc = .0
        best_val_f1 = .0
        min_valid_loss = float('Inf')

        for epoch in range(train_config['epochs']):
            # print('\n--------------- Epoch {} ---------------'.format(epoch + 1))

            # print('------------------------ train -----------------------')
            train_all_trues, train_all_scores, ave_train_loss, total_train_loss, train_acc, train_pre, train_rec, train_f1, train_auc, train_aupr = train_one_epoch(
                args=args,
                model=model,
                train_loader=train_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                threshold=threshold
            )
            # print('------------------------ val -----------------------')
            val_all_trues, val_all_scores, ave_val_loss, total_val_loss, val_acc, val_pre, val_rec, val_f1, val_auc, val_aupr = val_one_epoch(
                args=args,
                model=model,
                val_loader=val_loader,
                loss_func=loss_func,
                threshold=threshold,
            )

            # # save the best model by min loss
            # if ave_val_loss < min_valid_loss:
            #     count = 0
            #     min_valid_loss = ave_val_loss
            #     best_model = model
            #     best_val_auc = val_auc
            #     best_val_aupr = val_aupr
            #
            #     print("Found new best model. Min ave val loss is:{:.6f}   Validation AUC is: {:.6f}. ".format(ave_val_loss, val_auc))
            #
            # else:
            #     count += 1
            #     if count >= patience:
            #         torch.save(best_model.state_dict(), os.path.join(args.output_base_path, 'model_{}_{:.3f}_{:.3f}.pth'.format(i + 1, best_val_auc, best_val_aupr)))
            #         print(f'Fold {i + 1} training done!!!\n')
            #         break

            # Save the model by f1
            if val_f1 > best_val_f1:
                count = 0
                best_model = model
                best_val_auc = val_auc
                best_val_aupr = val_aupr
                best_val_f1 = val_f1

            else:
                count += 1
                if count >= patience:
                    # torch.save(best_model.state_dict(), os.path.join(args.output_base_path, 'model_{}_{:.3f}_{:.3f}.pth'.format(i + 1, best_val_auc, best_val_aupr)))
                    print(f'Fold {i + 1} training done!!!\n')
                    break

        kfold_val_auprs.append(best_val_aupr)
        kfold_val_aucs.append(best_val_auc)

        # print('------------------------ test -----------------------')
        test_all_gene, test_all_trues, test_all_scores, ave_test_loss, total_test_loss, test_acc, test_pre, test_rec, test_f1, test_auc, test_aupr = val_one_epoch(
            args=args,
            model=best_model,
            val_loader=test_loader,
            loss_func=loss_func,
            threshold=threshold,
            test_mode=True
        )

        kfold_test_trues.append(test_all_trues)    # copy test_all_trues k times. Check to see if the test_trues are the same for each cv, and check the code if they are different
        kfold_test_scores.append(test_all_scores)  # Record the predictions(sigmoid) of the test set in all cvs
        # kfold_test_aucs.append(test_auc)
        # kfold_test_auprs.append(test_aupr)

        np.save(os.path.join(args.output_base_path, 'test_all_trues.npy'), np.array(test_all_trues))        # Save labels on the test set
        np.save(os.path.join(args.output_base_path, 'kfold_test_scores.npy'), np.array(kfold_test_scores))  # Save predictions on the test set

    print(f'Finish training.\n')

    # for i, (test_auc, test_aupr) in enumerate(zip(kfold_test_aucs, kfold_test_auprs)):
    #     print('Fold {}: test AUC:{:.3f}   test AUPR:{:.3f}'.format(i+1, test_auc, test_aupr))

    # Average kfold models' results
    final_test_scores = np.sum(np.array(kfold_test_scores), axis=0) / kfold
    final_test_metrics = metrics.compute_metrics(test_all_trues, final_test_scores, threshold)[:]
    metrics.print_metrics('Final test', final_test_metrics)

    pre_test = pd.DataFrame(final_test_scores, columns=['score'], index=test_all_gene)
    pre_test['true label'] = test_all_trues.astype(int)
    pre_test.sort_values(by=['score'], inplace=True, ascending=False)
    pre_test.to_csv(path_or_buf=os.path.join(args.output_base_path, 'predicted_scores_test.txt'), sep='\t', index=True, header=True)
    pre_test_sort_axis0 = pre_test.sort_index()
    pre_test_sort_axis0.to_csv(path_or_buf=os.path.join(args.output_base_path, 'predicted_scores_sortname_test.txt'), sep='\t', index=True, header=True)
