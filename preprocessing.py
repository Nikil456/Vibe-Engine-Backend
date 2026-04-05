import pandas as pd

print("Loading yelp business data...")
df_biz = pd.read_json('yelp_json/yelp_dataset/yelp_academic_dataset_business.json', lines=True)

print("\nAvailable cities include: Philadelphia, Tampa, Nashville, New Orleans, Santa Barbara, etc.")
user_choice = input("Enter a city name (or type 'all' to process everything): ").strip()

if user_choice.lower() == 'all':
    print("\nWARNING: Processing the entire dataset. This will take a while and may require a lot of RAM!")
    df_restaurants = df_biz[df_biz['categories'].str.contains('Restaurants', na=False)]
    target_city_name = "All_Cities"
else:
    df_restaurants = df_biz[
        (df_biz['city'].str.lower() == user_choice.lower()) & 
        (df_biz['categories'].str.contains('Restaurants', na=False))
    ]
    target_city_name = user_choice

df_restaurants = df_restaurants[['business_id', 'name', 'latitude', 'longitude', 'review_count']]

target_biz_ids = set(df_restaurants['business_id']) #bad idea for large dataset
print(f"Filtered down to {len(target_biz_ids)} restaurants for {target_city_name}.")

if len(target_biz_ids) == 0:
    print("No restaurants found! The city might not be in the Yelp Academic Dataset. Exiting.")
    exit()

print("\nProcessing reviews in chunks... grab a coffee, this takes a minute.")
saved_reviews = []

chunk_iterator = pd.read_json('yelp_json/yelp_dataset/yelp_academic_dataset_review.json', lines=True, chunksize=20000)

for chunk in chunk_iterator:
    valid_reviews = chunk[chunk['business_id'].isin(target_biz_ids)]
    
    valid_reviews = valid_reviews[['business_id', 'text']]
    saved_reviews.append(valid_reviews)

df_all_reviews = pd.concat(saved_reviews, ignore_index=True)
print(f"Extracted {len(df_all_reviews)} relevant reviews.")

print("\nCleaning text for NLP pipeline...")
df_all_reviews = df_all_reviews.dropna(subset=['text'])

df_all_reviews['word_count'] = df_all_reviews['text'].apply(lambda x: len(str(x).split()))
df_clean_reviews = df_all_reviews[df_all_reviews['word_count'] >= 10].copy()

final_dataset = pd.merge(df_clean_reviews, df_restaurants, on='business_id', how='inner')
final_dataset = final_dataset.drop(columns=['word_count'])

output_filename = f"clean_{target_city_name.replace(' ', '_').lower()}_vibe_data.csv"
final_dataset.to_csv(output_filename, index=False)
print("\nSUCCESS! Data saved to {}".format(output_filename))