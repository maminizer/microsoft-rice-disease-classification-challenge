import CFG
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold



def main():

    train_df = pd.read_csv(
        '../input/microsoft-rice-disease-classification-challenge/train_df.csv')
    test = pd.read_csv(
        '../input/microsoft-rice-disease-classification-challenge/Test.csv')
    submission = pd.read_csv(
        '../input/microsoft-rice-disease-classification-challenge/SampleSubmission.csv')

    def get_path(image_id):
        return f"../input/microsoft-rice-disease-classification-challenge/Images/{image_id}"


    train_df["image_paths"] = train_df["Image_id"].apply(get_path)

    # remove RGN images
    train_df.drop(train_df[train_df.index % 2 != 0].index, inplace=True)
    train_df = train_df.reset_index(drop=True)

    # ====================================================
    # Label Encoder
    # ====================================================
    le = preprocessing.LabelEncoder()
    le.fit(train_df['Label'])
    train_df['Label'] = le.transform(train_df['Label'])

    # ====================================================
    # CV schem
    # ====================================================
    skf = StratifiedKFold(n_splits=CFG.nfolds, shuffle=True, random_state=CFG.seed)
    for fold, (trn_idx, vld_idx) in enumerate(skf.split(train_df, train_df[CFG.target_col])):
        train_df.loc[vld_idx, "folds"] = int(fold)
    train_df["folds"] = train_df["folds"].astype(int)
    train_df.to_csv('train.csv')
