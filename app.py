import streamlit as st
import torch
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Food Review Analyser",
    page_icon="🛒",
    layout="centered"
)

# ── Load models (cached so they don't reload on every interaction) ──


@st.cache_resource
def load_models():
    tokenizer = DistilBertTokenizer.from_pretrained('fionaghosh/amazon-food-review-sentiment')
    model     = DistilBertForSequenceClassification.from_pretrained('fionaghosh/amazon-food-review-sentiment')
    model.eval()
    embedder  = SentenceTransformer('all-MiniLM-L6-v2')
    return tokenizer, model, embedder
@st.cache_data
def load_topic_data():
    base_dir = Path(__file__).parent.resolve()
    return pd.read_csv(base_dir / 'topic_sentiment.csv')
tokenizer, model, embedder = load_models()
topic_sentiment            = load_topic_data()

# ── Helper functions ──────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

TOPIC_KEYWORDS = {
    'Coffee':                    ['coffee', 'espresso', 'brew', 'roast', 'latte', 'cappuccino', 'mocha', 'decaf', 'ground', 'beans'],
    'Tea':                       ['tea', 'green tea', 'herbal', 'steep', 'leaves', 'chamomile', 'chai', 'oolong', 'earl grey', 'matcha'],
    'Dog Food & Treats':         ['dog', 'dogs', 'puppy', 'puppies', 'treat', 'paw', 'kibble', 'canine', 'pup', 'bone'],
    'Product/Shipping (Amazon)': ['shipping', 'arrived', 'amazon', 'delivery', 'order', 'package', 'price', 'seller', 'refund', 'damaged', 'late'],
    'Cat Food':                  ['cat', 'cats', 'kitten', 'kittens', 'feline', 'litter', 'meow', 'whiskers'],
    'Chocolate':                 ['chocolate', 'cocoa', 'cacao', 'dark chocolate', 'milk chocolate', 'truffle', 'brownie', 'choco'],
    'Candy & Gifts':             ['candy', 'candies', 'gift', 'sweet', 'sweets', 'basket', 'gummy', 'jelly', 'lollipop', 'sugar'],
    'Cereal & Fiber':            ['cereal', 'fiber', 'oat', 'oats', 'oatmeal', 'grain', 'bran', 'granola', 'flakes', 'muesli'],
    'Protein Bars':              ['protein', 'bar', 'bars', 'nutrition', 'energy bar', 'whey', 'supplement', 'fitness', 'workout'],
    'Cookies':                   ['cookie', 'cookies', 'biscuit', 'biscuits', 'cracker', 'crackers', 'baked', 'shortbread', 'wafer'],
    'Fruits':                    ['fruit', 'fruits', 'apple', 'banana', 'mango', 'berry', 'berries', 'strawberry', 'blueberry', 'raspberry',
                                  'orange', 'lemon', 'lime', 'grape', 'grapes', 'peach', 'pear', 'plum', 'cherry', 'cherries',
                                  'pineapple', 'watermelon', 'melon', 'kiwi', 'dried fruit', 'raisin', 'raisins', 'apricot'],
    'Vegetables':                ['vegetable', 'vegetables', 'veggie', 'veggies', 'spinach', 'kale', 'broccoli', 'carrot', 'carrots',
                                  'tomato', 'tomatoes', 'onion', 'onions', 'garlic', 'pepper', 'peppers', 'celery', 'lettuce',
                                  'cucumber', 'zucchini', 'potato', 'potatoes', 'sweet potato', 'corn', 'pea', 'peas',
                                  'mushroom', 'mushrooms', 'asparagus', 'cauliflower', 'cabbage', 'beetroot', 'beet'],
    'Meat & Chicken':            ['chicken', 'beef', 'meat', 'pork', 'turkey', 'lamb', 'steak', 'bacon', 'sausage', 'sausages',
                                  'ham', 'salami', 'pepperoni', 'brisket', 'ribeye', 'ground beef', 'mince', 'jerky',
                                  'poultry', 'drumstick', 'wing', 'wings', 'breast', 'thigh', 'roast'],
    'Fish & Seafood':            ['fish', 'salmon', 'tuna', 'cod', 'tilapia', 'sardine', 'sardines', 'anchovy', 'anchovies',
                                  'shrimp', 'prawns', 'lobster', 'crab', 'scallop', 'scallops', 'oyster', 'oysters',
                                  'seafood', 'clam', 'clams', 'mussel', 'mussels', 'squid', 'calamari', 'trout', 'halibut'],
    'Nuts & Seeds':              ['nut', 'nuts', 'almond', 'almonds', 'cashew', 'cashews', 'walnut', 'walnuts', 'peanut', 'peanuts',
                                  'pistachio', 'pistachios', 'pecan', 'pecans', 'seed', 'seeds', 'sunflower', 'pumpkin seed',
                                  'chia', 'flaxseed', 'hemp', 'macadamia', 'hazelnut', 'hazelnuts'],
    'Dairy & Eggs':              ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'yoghurt', 'egg', 'eggs', 'dairy',
                                  'cheddar', 'mozzarella', 'parmesan', 'brie', 'whipped cream', 'ice cream', 'kefir'],
    'Pasta & Grains':            ['pasta', 'noodle', 'noodles', 'spaghetti', 'rice', 'quinoa', 'barley', 'wheat',
                                  'macaroni', 'penne', 'linguine', 'fettuccine', 'couscous', 'lentil', 'lentils',
                                  'chickpea', 'chickpeas', 'bean', 'beans', 'legume'],
    'Sauces & Condiments':       ['sauce', 'sauces', 'ketchup', 'mustard', 'mayo', 'mayonnaise', 'dressing', 'vinegar',
                                  'soy sauce', 'hot sauce', 'salsa', 'relish', 'marinade', 'seasoning', 'spice', 'spices',
                                  'herb', 'herbs', 'salt', 'pepper', 'chilli', 'sriracha', 'bbq'],
    'Snacks & Chips':            ['snack', 'snacks', 'chip', 'chips', 'crisp', 'crisps', 'popcorn', 'pretzel', 'pretzels',
                                  'puff', 'puffs', 'rice cake', 'trail mix', 'crackers', 'dip', 'nacho'],
    'Beverages & Juice':         ['juice', 'drink', 'drinks', 'beverage', 'beverages', 'smoothie', 'lemonade', 'soda',
                                  'sparkling', 'water', 'coconut water', 'energy drink', 'sports drink', 'kombucha'],
    'Oils & Vinegars':           ['oil', 'oils', 'olive oil', 'coconut oil', 'avocado oil', 'vegetable oil', 'vinegar',
                                  'balsamic', 'apple cider vinegar', 'sesame oil', 'ghee'],
}

def guess_topic(text):
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'Uncategorised'

def predict_sentiment(text):
    cleaned = clean_text(text)
    inputs  = tokenizer(
        cleaned,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1).squeeze()
    neg_prob = probs[0].item()
    pos_prob = probs[1].item()
    label    = 'Positive' if pos_prob > neg_prob else 'Negative'
    return label, pos_prob, neg_prob

# ── UI ────────────────────────────────────────────────────────
st.title("🛒 Amazon Food Review Analyser")
st.markdown("Paste any Amazon food review below to get sentiment prediction and category insights.")

review_text = st.text_area(
    "Paste your review here:",
    height=180,
    placeholder="e.g. I've been buying this coffee for years. The flavour is rich and consistent..."
)

if st.button("Analyse Review", type="primary"):
    if not review_text.strip():
        st.warning("Please paste a review first.")
    else:
        with st.spinner("Analysing..."):
            label, pos_prob, neg_prob = predict_sentiment(review_text)
            topic                     = guess_topic(review_text)

        # ── Sentiment result ──
        col1, col2, col3 = st.columns(3)
        with col1:
            colour = "🟢" if label == "Positive" else "🔴"
            st.metric("Sentiment", f"{colour} {label}")
        with col2:
            st.metric("Positive confidence", f"{pos_prob*100:.1f}%")
        with col3:
            st.metric("Negative confidence", f"{neg_prob*100:.1f}%")

        st.markdown("---")

        # ── Topic ──
        st.subheader(f" Detected category: {topic}")

        # ── Chart: how does this topic compare ──
        if topic != 'Uncategorised':
            st.markdown("**How does this category compare in negative reviews?**")
            fig, ax = plt.subplots(figsize=(8, 4))
            colours = [
                '#d62728' if t == topic else '#aec7e8'
                for t in topic_sentiment['topic_label']
            ]
            ax.barh(topic_sentiment['topic_label'],
                    topic_sentiment['negative_pct'],
                    color=colours)
            ax.axvline(
                x=topic_sentiment['negative_pct'].mean(),
                color='black', linestyle='--', alpha=0.5, label='Average'
            )
            ax.set_xlabel('Negative Review %')
            ax.set_title('Negative Review Rate by Category')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            row = topic_sentiment[topic_sentiment['topic_label'] == topic]
            if not row.empty:
                neg_pct = row['negative_pct'].values[0]
                avg     = topic_sentiment['negative_pct'].mean()
                diff    = neg_pct - avg
                if diff > 0:
                    st.info(f"**{topic}** has a {neg_pct}% negative review rate — "
                            f"{abs(diff):.1f}% above the category average.")
                else:
                    st.success(f"**{topic}** has a {neg_pct}% negative review rate — "
                               f"{abs(diff):.1f}% below the category average.")

st.markdown("---")
st.caption("Built with DistilBERT + BERTopic · Amazon Fine Food Reviews dataset · Fiona Ghosh")