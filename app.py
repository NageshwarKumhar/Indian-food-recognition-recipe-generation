import streamlit as st
from PIL import Image
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch

st.set_page_config(
    page_title="Indian Food AI",
    page_icon="🍛",
    layout="wide"
)

st.title("🍛 Indian Food Recognition & Recipe Generation")
st.write(
    "Upload a food image to predict its category, view regional information, "
    "and generate a short description and typical recipe."
)

CLASSIFIER_MODEL = "Subhash5/indian-food-classifier"
TEXT_MODEL = "google/flan-t5-small"

# The classifier supports 20 categories. Dalma and chhena poda are included
# in the recipe knowledge base but are not classes in this pretrained model.
FOOD_INFO = {
    "masala_dosa": {
        "name": "Masala Dosa",
        "region": "South India, especially Karnataka and Tamil Nadu",
        "description": "A thin fermented rice-and-lentil crepe filled with spiced potato masala.",
        "recipe": "Prepare a fermented rice-and-urad-dal batter. Spread it thinly on a hot greased pan, cook until crisp, add spiced potato filling, fold, and serve with chutney and sambar."
    },
    "idli": {
        "name": "Idli",
        "region": "South India",
        "description": "A soft, steamed savoury cake made from fermented rice and urad-dal batter.",
        "recipe": "Soak rice and urad dal separately, grind, mix and ferment overnight. Pour into greased idli moulds and steam until cooked. Serve with sambar and chutney."
    },
    "dhokla": {
        "name": "Dhokla",
        "region": "Gujarat, western India",
        "description": "A soft, savoury steamed cake commonly made with fermented or leavened gram-flour batter.",
        "recipe": "Mix gram flour with water, lemon juice, salt and a raising agent. Steam the batter until set. Pour over a tempering of mustard seeds, curry leaves and green chilli; cool and cut into pieces."
    },
    "biryani": {
        "name": "Biryani",
        "region": "Found across India; regional styles include Hyderabadi, Lucknowi and Kolkata",
        "description": "A layered or cooked-together rice dish seasoned with aromatic spices, herbs and vegetables or meat.",
        "recipe": "Cook basmati rice until partly done. Prepare a spiced vegetable or meat masala with onion, ginger, garlic and whole spices. Layer with rice, herbs and a little saffron milk if desired, then cover and cook on low heat until fragrant and fully cooked."
    },
    "dalma": {
        "name": "Dalma",
        "region": "Odisha, eastern India",
        "description": "A traditional Odia dish of lentils cooked with vegetables and gently spiced.",
        "recipe": "Cook toor dal with chopped vegetables such as pumpkin, raw banana and papaya, plus turmeric and salt. Prepare a tempering with ghee or oil, cumin, dried red chilli and optional roasted spice powder. Mix and simmer until the vegetables are tender."
    },
    "chhena poda": {
        "name": "Chhena Poda",
        "region": "Odisha, eastern India",
        "description": "A traditional Odia baked dessert made from fresh chhena, sugar and flavourings, with a caramelised crust.",
        "recipe": "Crumble fresh chhena and mix with sugar, a little semolina, cardamom and optional nuts. Transfer to a greased baking tin and bake until set with a deep golden-brown crust. Cool before slicing."
    },
    "samosa": {
        "name": "Samosa",
        "region": "Popular throughout India; regional variations are common",
        "description": "A crisp pastry snack often filled with spiced potatoes and peas.",
        "recipe": "Prepare a firm pastry dough. Make a filling with boiled potatoes, peas and spices. Shape the dough into cones, fill and seal, then fry on medium-low heat until crisp and golden."
    },
    "chole_bhature": {
        "name": "Chole Bhature",
        "region": "North India, especially Punjab and Delhi",
        "description": "A popular combination of spiced chickpea curry and deep-fried leavened bread.",
        "recipe": "Cook soaked chickpeas until tender. Simmer with onion, tomato, ginger, garlic and chole spices. Prepare a soft leavened dough, roll into discs and deep-fry bhature until puffed."
    },
    "pav_bhaji": {
        "name": "Pav Bhaji",
        "region": "Mumbai, Maharashtra, western India",
        "description": "A spiced mashed vegetable curry served with butter-toasted bread rolls.",
        "recipe": "Boil potatoes, peas and mixed vegetables. Mash and cook with onion, tomato, capsicum and pav bhaji masala. Finish with lemon and coriander; toast pav with butter and serve."
    },
    "jalebi": {
        "name": "Jalebi",
        "region": "Popular across India, with regional variations",
        "description": "A spiral-shaped sweet made from batter and soaked in sugar syrup.",
        "recipe": "Prepare a pourable batter and allow it to ferment if using a traditional method. Pipe spirals into medium-hot oil and fry until crisp. Briefly soak in warm sugar syrup flavoured with cardamom or saffron."
    },
    "dosa": {
        "name": "Dosa",
        "region": "South India",
        "description": "A thin crepe made from fermented rice and lentil batter.",
        "recipe": "Spread fermented rice-and-urad-dal batter thinly on a hot pan. Cook until the underside is golden and crisp, then fold and serve with chutney or sambar."
    },
    "idli": {
        "name": "Idli",
        "region": "South India",
        "description": "A soft, steamed savoury cake made from fermented rice and urad-dal batter.",
        "recipe": "Soak and grind rice and urad dal, mix and ferment overnight. Steam in idli moulds until fluffy and cooked through. Serve with sambar and chutney."
    },

    "Paani Puri": {
        "name": "Paani Puri",
        "region": "Western, Northern and Eastern India; especially popular in Maharashtra, Gujarat, Uttar Pradesh and Bihar.",
        "description": (
            "A popular Indian street food consisting of crisp, hollow puris "
            "filled with spiced potato or chickpeas and tangy, flavoured water. "
            "It is also known as golgappa or puchka in different regions."
        ),
        "recipe": (
            "Ingredients: pani puris, boiled potatoes, cooked chickpeas, "
            "mint, coriander, green chillies, tamarind, roasted cumin, "
            "black salt and chilled water. "
            "Method: Prepare a tangy, spicy water by blending mint, coriander, "
            "green chillies, tamarind and spices with water. Mash the potatoes "
            "and mix with chickpeas and seasoning. Make a small hole in each "
            "puri, fill it with the potato mixture, add the flavoured water, "
            "and serve immediately."
        )
    },
}

def normalize_label(label: str) -> str:
    label = label.lower().replace("-", "_").strip()
    aliases = {
        "masala dosa": "masala_dosa",
        "kaathi rolls": "kaathi_rolls",
        "chole bhature": "chole_bhature",
        "pav bhaji": "pav_bhaji",
        "dal makhani": "dal_makhani",
        "butter naan": "butter_naan",
        "chhena poda": "chhena poda",
    }
    return aliases.get(label.replace("_", " "), label)

@st.cache_resource(show_spinner="Loading image classifier (first run may take a few minutes)...")
def load_classifier():
    return pipeline(
        task="image-classification",
        model=CLASSIFIER_MODEL,
        device=0 if torch.cuda.is_available() else -1
    )


@st.cache_resource
def load_text_generator():
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    return tokenizer, model



def generate_text(food_name, region, description, recipe):
    tokenizer, model = load_text_generator()

    prompt = (
        f"Describe {food_name}, a popular Indian food. "
        f"Regional association: {region}. "
        f"Description: {description}. "
        f"Typical preparation: {recipe}. "
        "Write a short food description and numbered cooking steps."
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=2,
            early_stopping=True,
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()

    if not result:
        raise RuntimeError(
            "FLAN-T5 returned empty text. Check the model output and Transformers version."
        )

    return result



uploaded = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Classifying the image..."):
            classifier = load_classifier()
            predictions = classifier(image, top_k=5)

        with right:
            st.subheader("Model predictions")
            for item in predictions:
                st.write(f"**{item['label'].replace('_', ' ').title()}** — {item['score']:.1%}")
                st.progress(min(max(float(item["score"]), 0.0), 1.0))

        top = predictions[0]
        raw_label = top["label"]
        key = normalize_label(raw_label)
        default_name = raw_label.replace("_", " ").title()

        # Let the user correct the label, especially for regional foods
        # not represented in the pretrained model's 20 classes.
        st.subheader("Confirm the food")
        choices = [
            "Use model prediction",
            "Dosa",
            "Idli",
            "Dhokla",
            "Dalma",
            "Chhena Poda",
            "Biryani",
            "Samosa",
            "Chole Bhature",
            "Pav Bhaji",
            "Jalebi",
            "Other / not sure"
        ]
        selected = st.selectbox("If needed, correct the predicted food:", choices)

        if selected == "Use model prediction":
            food_key = key
            food_name = FOOD_INFO.get(food_key, {}).get("name", default_name)
            confidence = float(top["score"])
        elif selected == "Other / not sure":
            food_key = None
            food_name = "Unconfirmed food"
            confidence = float(top["score"])
        else:
            food_name = selected
            food_key = normalize_label(selected)
            confidence = float(top["score"])

        st.caption(
            "Confidence is the classifier's score for its top predicted class. "
            "It is not a guarantee that the prediction is correct."
        )

        if food_key and food_key in FOOD_INFO:
            info = FOOD_INFO[food_key]
            st.divider()
            st.subheader(f"🍽️ {info['name']}")
            st.write(f"**Likely regional association:** {info['region']}")
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Typical recipe:** {info['recipe']}")

            if st.button("Generate description and recipe with FLAN-T5"):
                with st.spinner("Generating text..."):
                    try:
                        generated = generate_text(
                            info["name"], info["region"], info["description"], info["recipe"]
                        )
                        st.subheader("Generated text (FLAN-T5)")
                        st.write(generated)
                    except Exception as exc:
                        st.error("Text generation could not run. The reference description and recipe are still available.")
                        st.exception(exc)
        elif selected == "Other / not sure":
            st.warning(
                "The app cannot confidently provide a supported recipe for this selection. "
                "Choose one of the listed dishes to see its regional information and typical recipe."
            )
        else:
            st.warning(
                f"The pretrained classifier predicted **{food_name}**, but this food is not yet in the recipe knowledge base. "
                "Add its regional facts and recipe to FOOD_INFO in app.py."
            )

        with st.expander("Project model details and limitations"):
            st.markdown(
                f"""
                - **Image model:** `{CLASSIFIER_MODEL}` (a pretrained Indian-food image classifier).
                - **Text model:** `{TEXT_MODEL}` (FLAN-T5-small).
                - The pretrained image model has a fixed set of classes; it does not directly recognize every Indian dish.
                - Dalma and Chhena Poda are included as recipe knowledge entries, but are not supported classifier classes in this initial version. Selecting them manually demonstrates the recipe-generation part, not automatic visual recognition of those dishes.
                - Regional association is a curated knowledge-base entry, not a visual prediction.
                - Recipe text is illustrative. Verify ingredients, cooking times and food safety before use.
                """
            )
    except Exception as exc:
        st.error("Could not process this image. Try a valid JPG or PNG file.")
        st.exception(exc)
else:
    st.info("Upload a food photo to begin.")
    st.markdown(
        "**Try images of:** dosa, idli, dhokla, samosa, chole bhature or pav bhaji. "
    )
