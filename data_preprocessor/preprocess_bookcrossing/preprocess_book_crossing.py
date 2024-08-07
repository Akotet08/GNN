import json
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and split Book-Crossing dataset")
    parser.add_argument('--data_path_json', type=str, default='../dataset_path.json',
                        help='Path to JSON file containing data paths')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Proportion of the data to include in the train split')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--silent', action='store_true', help='Suppress print statements if set')
    parser.add_argument('--implicit_bottom', type=float, default=None, help="change rating to implicit feedback")
    parser.add_argument('--sample_ratio', type=float, default=0.4, help="ratio for sampling for efficiency")
    return parser.parse_args()


def encode_features(interaction_df, item_feature_df):
    user_embedder = LabelEncoder()
    item_embedder = LabelEncoder()

    missing_items_isbn = set(interaction_df['ISBN']).difference(set(item_feature_df['ISBN']))
    interaction_df = interaction_df[~interaction_df['ISBN'].isin(missing_items_isbn)]

    interaction_df['user_id'] = user_embedder.fit_transform(interaction_df['User-ID'])
    item_feature_df['item_id'] = item_embedder.fit_transform(item_feature_df['ISBN'])

    interaction_df['item_id'] = item_embedder.transform(interaction_df['ISBN'])

    interaction_df.drop(['ISBN', 'User-ID'], axis=1, inplace=True)
    item_feature_df.drop(['ISBN'], axis=1, inplace=True)

    item_feature_df['Year-Of-Publication'] = item_feature_df['Year-Of-Publication'].astype(int)

    # Encode string features in item feature dataframe
    str_features = ['Book-Title', 'Book-Author', 'Publisher']
    for col in str_features:
        encoder = LabelEncoder()
        item_feature_df[col] = encoder.fit_transform(item_feature_df[col])

    return interaction_df, item_feature_df


def split_train_test(interaction_df, args):
    interaction = interaction_df.sample(frac=args.sample_ratio, random_state=args.seed).reset_index(drop=True)

    config = {'num_users': interaction['user_id'].nunique(),
              'num_edges': len(interaction)}

    # Initial train-test split
    split_index = int(args.train_split * len(interaction))
    interact_train = interaction[:split_index]
    interact_test = interaction[split_index:]

    # Ensure all users are in the training set
    train_users = set(interact_train['user_id'])
    test_users = set(interact_test['user_id'])

    missing_users = test_users - train_users

    # Transfer missing users' interactions to training set
    for user in missing_users:
        user_interactions = interact_test[interact_test['user_id'] == user]
        interact_train = pd.concat([interact_train, user_interactions.head(1)], ignore_index=True)
        interact_test = interact_test.drop(user_interactions.index[:1])

    return interact_train, interact_test, config

def main():
    args = parse_arguments()

    # Load paths from JSON configuration
    with open(args.data_path_json, 'r') as f:
        paths = json.load(f)

    raw_data_path = paths['book_crossing']['raw']
    processed_data_path = paths['book_crossing']['processed']

    # Load and process item feature data (books)
    item_feature_df = pd.read_csv(f'{raw_data_path}/BX_Books.csv', delimiter=';', encoding='ISO-8859-1')
    item_feature_df = item_feature_df.drop(['Image-URL-M', 'Image-URL-S', 'Image-URL-L'], axis=1)
    interaction = pd.read_csv(f'{raw_data_path}/BX-Book-Ratings.csv', delimiter=';', encoding='ISO-8859-1')

    # Rename columns for consistency
    interaction.rename(columns={'Book-Rating': 'rating'}, inplace=True)
    interaction, item_feature_df = encode_features(interaction, item_feature_df)

    if args.implicit_bottom is not None:
        interaction['rating'] = interaction['rating'] > args.implicit_bottom
    interaction['rating'] = interaction['rating'].astype(int)

    for col in item_feature_df.columns:
        if col not in ['item_id']:
            scaler = StandardScaler()
            item_feature_df[col] = scaler.fit_transform(item_feature_df[col].to_numpy().reshape(-1, 1))

    interact_train, interact_test, config = split_train_test(interaction, args)

    config['num_items'] = item_feature_df['item_id'].nunique()

    suffix = args.sample_ratio
    interact_train.to_csv(f'{processed_data_path}/book_crossing_train_{suffix}.csv', index=False)
    interact_test.to_csv(f'{processed_data_path}/book_crossing_test_{suffix}.csv', index=False)
    item_feature_df.to_csv(f'{processed_data_path}/book_crossing_item_features.csv', index=False)

    with open(f'{processed_data_path}/book_crossing_config.json', 'w') as f:
        json.dump(config, f)

    if not args.silent:
        print(f'Train set size: {len(interact_train)} saved to', f'{processed_data_path}/book_crossing_train_{suffix}.csv')
        print(f'Test set size: {len(interact_test)} saved to', f'{processed_data_path}/book_crossing_test_{suffix}.csv')
        print(f'config: {config} \n saved to', f'{processed_data_path}/book_crossing_config.json')
        print('========> Saved Train and Test and config files')


if __name__ == '__main__':
    main()
