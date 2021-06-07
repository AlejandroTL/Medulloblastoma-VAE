import shap
from captum.attr import DeepLiftShap
import numpy as np
import pandas as pd
import argparse
import torch
import torch.utils.data
import general
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost
from collections import Counter
from datetime import datetime


def aux_counter(lista, names, dim):
    total_list = []

    for i_i in range(len(lista)):
        prov = []
        for j in range(dim):
            d = Counter(lista[i_i])
            if j in lista[i_i]:
                prov.append(d[j])
            else:
                prov.append(0)
        total_list.append(prov)
    total = pd.DataFrame(total_list, columns=names)
    total_sum = total.sum(axis=0)
    return pd.DataFrame(total_sum)


def xgboost_preprocessing(train_dataset, colors_trainset, test_dataset, colors_testset):
    # Tensor2DF
    train_df = pd.DataFrame(train_dataset.detach().numpy())
    test_df = pd.DataFrame(test_dataset.detach().numpy())

    # Create all dataset
    entire_dataset = pd.concat((train_df, test_df), ignore_index=True)
    entire_colors = np.concatenate((colors_trainset, colors_testset))

    # Split on groups and clean outliers
    entire_shh = entire_dataset.loc[entire_colors == 'SHH']
    entire_wnt = entire_dataset.loc[entire_colors == 'WNT']
    entire_g3 = entire_dataset.loc[entire_colors == 'Group3']
    entire_g4 = entire_dataset.loc[entire_colors == 'Group4']

    outliers_shh = find_outliers(entire_shh, 0.1)
    outliers_wnt = find_outliers(entire_wnt, 0.1)
    outliers_g3 = find_outliers(entire_g3, 0.1)
    outliers_g4 = find_outliers(entire_g4, 0.1)
    outliers_total = outliers_wnt.union(outliers_shh.union(outliers_g3.union(outliers_g4)))

    entire_data_outliers = entire_dataset.drop(list(outliers_total))
    entire_colors_outliers = pd.DataFrame(entire_colors).drop(list(outliers_total))

    entire_data_outliers = torch.tensor(entire_data_outliers.values).float()
    entire_colors_outliers = entire_colors_outliers.to_numpy().reshape(len(entire_colors_outliers))

    x_train, x_test, y_train, y_test = train_test_split(entire_data_outliers, entire_colors_outliers,
                                                        test_size=0.3,
                                                        random_state=1)

    return x_train, x_test, y_train, y_test


def get_embeddings(model, dataloader, device):
    model.eval()
    rec_model = np.zeros(shape=(0, model.decoder[2].out_features))
    embedding_model = np.zeros(shape=(0, features))
    mean_model = np.zeros(shape=(0, features))
    logvar_model = np.zeros(shape=(0, features))
    with torch.no_grad():  # in validation we don't want to update weights
        for data in dataloader:
            data = data.to(device)
            reconstruction, mean, logvar, coded = model(data)
            rec_model = np.concatenate((rec_model, reconstruction), axis=0)
            mean_model = np.concatenate((mean_model, mean), axis=0)
            logvar_model = np.concatenate((logvar_model, logvar), axis=0)
            embedding_model = np.concatenate((embedding_model, coded), axis=0)

    return rec_model, embedding_model, mean_model, logvar_model


def xgboost_shap(model, train_loader_beta, colors_coded_train,
                 test_loader_beta, colors_coded_test, latent_variables, device):
    # Get the embeddings
    _, coded_train_beta, _, _ = get_embeddings(model, train_loader_beta, device)
    _, coded_test_beta, _, _ = get_embeddings(model, test_loader_beta, device)

    print("Inside :", coded_train_beta.shape, len(colors_coded_train.ravel()))

    # Train the classifier
    xgboost_classifier_beta = xgboost.XGBClassifier(random_state=123)
    xgboost_classifier_beta.fit(coded_train_beta, colors_coded_train.ravel())

    y_pred = xgboost_classifier_beta.predict(coded_test_beta)

    accuracy = accuracy_score(colors_coded_test.ravel(), y_pred)
    print("Accuracy with all LV: ", accuracy)

    # SHAP Tree Explainer to get the importance
    whole_dataset = np.concatenate((coded_train_beta, coded_test_beta))
    explainer = shap.TreeExplainer(xgboost_classifier_beta)
    shap_values = explainer.shap_values(whole_dataset)  # we want to explain the whole dataset

    feature_importance_shap = np.sum(np.abs(shap_values).mean(1),
                                     axis=0)  # mean within class and summation between classes
    indices_beta_sort = np.argsort(feature_importance_shap)[::-1]

    # Top features
    indices_beta = indices_beta_sort[:latent_variables]

    coded_beta_filtered = pd.DataFrame(coded_train_beta)[indices_beta]
    coded_beta_filtered_test = pd.DataFrame(coded_test_beta)[indices_beta]

    xgboost_classifier_beta = xgboost.XGBClassifier()
    xgboost_classifier_beta.fit(coded_beta_filtered, colors_coded_train.ravel())

    y_pred = xgboost_classifier_beta.predict(coded_beta_filtered_test)

    accuracy = accuracy_score(colors_coded_test, y_pred)
    print("Accuracy with X LV: ", accuracy)

    return indices_beta


def shap_vae(model, examples, shap_indices):

    background = examples[:150]
    test_shap = examples[150:]

    net = model.encoder
    dl = DeepLiftShap(net, True)
    genes = []
    for ind in shap_indices:
        attribution = dl.attribute(test_shap, baselines=background, target=int(ind))
        np_attribution = attribution.detach().numpy()
        for element in range(len(np_attribution.mean(0))):
            if np_attribution.mean(0)[element] > np_attribution.mean(0).mean() + np_attribution.mean(0).std() * 3:
                genes.append(element)
            if np_attribution.mean(0)[element] < np_attribution.mean(0).mean() - np_attribution.mean(0).std() * 3:
                genes.append(element)

    return genes


def find_outliers(data, percentage=0.10):
    candidates_final = []
    return_list = []

    for column in data.columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        candidates_final = candidates_final + (list(data.loc[data[column] < (q1 - 1.5 * iqr)].index))
        candidates_final = candidates_final + (list(data.loc[data[column] > (q3 + 1.5 * iqr)].index))

    for i in candidates_final:
        if candidates_final.count(i) > percentage * len(data.columns):
            return_list.append(i)

    return set(return_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # VAE Definition
    parser.add_argument("--hidden_layer", help="hidden layer dimension")
    parser.add_argument("--features", help="bottleneck dimension")

    # Setup options
    parser.add_argument("--loss", help="loss function", choices=["bce", "mse"], default="mse")
    parser.add_argument("--lr", help="learning rate", default=0.0001)

    # Training options
    parser.add_argument("--epochs", help="epochs per cycle")
    parser.add_argument("--cycles", help="number of cycles")
    parser.add_argument("--initial_width", help="initial width of beta=0")
    parser.add_argument("--reduction", help="reduction of width per cycle")
    parser.add_argument("--beta", help="beta")

    # Data
    parser.add_argument("--train_path", help="path to training csv")
    parser.add_argument("--test_path", help="path to test csv")
    parser.add_argument("--batch_size", help="batch size", default=8)
    parser.add_argument("--colors_train", help="path to the train subgroups")
    parser.add_argument("--colors_test", help="path to the test subgroups")

    # Plots

    parser.add_argument("--plots", help="loss plots", choices=["0", "1"], default="0")

    # XGboost params

    parser.add_argument("--LV", help="number of LV selected", default=20)

    # Iterations

    parser.add_argument("--iterations", help="number of models to train")

    args = parser.parse_args()

    # input_dim = int(args.input_dim)
    mid_dim = int(args.hidden_layer)
    features = int(args.features)

    lr = float(args.lr)

    ch_epochs = int(args.epochs)
    ch_cycles = int(args.cycles)
    ch_width = int(args.initial_width)
    ch_reduction = int(args.reduction)
    ch_beta = float(args.beta)

    ch_batch_size = int(args.batch_size)

    ch_latent_variables = int(args.LV)

    ch_iterations = int(args.iterations)

    # Load and preprocess the data
    train_data, train_loader, test_data, test_loader, genes_name = general.data2tensor(args.train_path, args.test_path,
                                                                                       ch_batch_size)
    print("Data Preprocessing successfully")

    colors_train, colors_test = general.colors_preprocessing(args.colors_train, args.colors_test)

    shap_aux_list = []
    shap_aux_list_sets = []
    for iteration in range(ch_iterations):
        # Chosen model with dimensions
        chosen_model = general.VAE(input_dim=len(train_data[0]), mid_dim=mid_dim, features=features)

        # Training
        tr_l, tt_l, tr_kl, tt_kl, tr_r, tt_r, id_string, dev = general.cyclical_training(chosen_model, train_loader,
                                                                                         test_loader,
                                                                                         epochs=ch_epochs,
                                                                                         cycles=ch_cycles,
                                                                                         initial_width=ch_width,
                                                                                         reduction=ch_reduction,
                                                                                         beta=ch_beta,
                                                                                         option=args.loss,
                                                                                         learning_rate=lr)

        if args.plots == "1":
            general.loss_plots(tr_l, tt_l, tr_kl, tt_kl, tr_r, tt_r, id_string)

        # XGBoost pipeline

        # The train-test split is random. As we will perform the pipeline several times, I prefer
        # split it everytime
        xgboost_train, xgboost_test, xgboost_train_colors, xgboost_test_colors = xgboost_preprocessing(train_data,
                                                                                                       colors_train,
                                                                                                       test_data,
                                                                                                       colors_test)

        # We create a dataset to work with SHAP that it's all the dataset without outliers
        SHAP_dataset = torch.cat((xgboost_train, xgboost_test))

        # Create Dataloaders
        xgboost_train_dataloader = torch.utils.data.DataLoader(
            xgboost_train,
            batch_size=ch_batch_size,
            shuffle=False,
        )

        xgboost_test_dataloader = torch.utils.data.DataLoader(
            xgboost_test,
            batch_size=ch_batch_size,
            shuffle=False,
        )

        shap_dataloader = torch.utils.data.DataLoader(  # Dataloader for SHAP computations. Batch size higher!
            xgboost_train,
            batch_size=256,
            shuffle=True,
        )

        # SHAP Pipeline

        # We get a entire batch of the shap_dataloader, i.e., 256 genomic profiles
        profiles = next(iter(shap_dataloader))

        indices = xgboost_shap(chosen_model, xgboost_train_dataloader, xgboost_train_colors,
                               xgboost_test_dataloader, xgboost_test_colors, ch_latent_variables, dev)

        focus_genes = shap_vae(chosen_model, profiles, indices)

        shap_aux_list.append(focus_genes)
        shap_aux_list_sets.append(set(focus_genes))

    shap_df = aux_counter(shap_aux_list, genes_name, len(train_data[0]))
    shap_df_sets = aux_counter(shap_aux_list_sets, genes_name, len(train_data[0]))

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%H_%M")

    shap_df.to_csv(f"SHAP_Reports/SHAP_Report{dt_string}_{ch_iterations}.csv")
    shap_df_sets.to_csv(f"SHAP_Reports/SHAP_Report_Set{dt_string}_{ch_iterations}.csv")