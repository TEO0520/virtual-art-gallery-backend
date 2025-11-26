import os
from flask import Flask, jsonify, request
from supabase import create_client, Client
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import json
import google.generativeai as genai
from PIL import Image
import io

app = Flask(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')


def get_recommendations(user_id: str):
    try:
        likes_data = supabase.table('Like').select('customUserId, artworkId').execute()
        
        if not likes_data.data:
            return {"recommended_artwork": None, "recommended_artist": None}

        df = pd.DataFrame(likes_data.data)
        
        interaction_matrix = df.pivot_table(index='customUserId', columns='artworkId', aggfunc='size', fill_value=0)

        if user_id not in interaction_matrix.index:
            print(f"User {user_id} is a new user with no likes. No recommendations available.")
            return {"recommended_artwork": None, "recommended_artist": None}

        user_similarity = cosine_similarity(interaction_matrix)
        user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)

        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
        
        if similar_users.empty or similar_users.iloc[0] == 0:
            return {"recommended_artwork": None, "recommended_artist": None}

        current_user_likes = set(interaction_matrix.loc[user_id][interaction_matrix.loc[user_id] > 0].index)
        
        for similar_user_id in similar_users.index:
            similar_user_likes = set(interaction_matrix.loc[similar_user_id][interaction_matrix.loc[similar_user_id] > 0].index)
            
            recommended_artwork_ids = list(similar_user_likes - current_user_likes)
            
            if not recommended_artwork_ids:
                continue 

            artwork_details_response = supabase.table('Artwork').select('*, User!Artwork_customUserId_fkey(userName, customUserId)').in_('artworkId', recommended_artwork_ids).execute()
            if not artwork_details_response.data:
                continue

            for artwork in artwork_details_response.data:
                artwork_owner = artwork.get('User')
                if not artwork_owner:
                    continue
                artwork_owner_id = artwork_owner.get('customUserId')
                
                if artwork_owner_id != user_id:
                    recommended_artwork = artwork
                    
                    artist_details = supabase.table('artists_with_details').select('*').eq('customUserId', artwork_owner_id).single().execute()
                    recommended_artist = artist_details.data if artist_details.data else None
                    
                    return {"recommended_artwork": recommended_artwork, "recommended_artist": recommended_artist}
        
        return {"recommended_artwork": None, "recommended_artist": None}

    except Exception as e:
        print(f"CRITICAL Error in recommendation logic for user {user_id}: {e}")
        return {"recommended_artwork": None, "recommended_artist": None}



@app.route('/recommend', methods=['GET'])
def recommend_handler():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    recommendations = get_recommendations(user_id)
    return jsonify(recommendations)

@app.route('/analyze-artwork', methods=['POST'])
def analyze_artwork_handler():
    if 'artwork_image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['artwork_image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.stream.read()
        print(f">>> [SERVER LOG] Received image with size: {len(image_bytes)} bytes")
        
        img = Image.open(io.BytesIO(image_bytes))

        prompt = """
        Analyze the attached image. First, determine if it is a piece of art (like a painting, sculpture, digital art, etc.) or just a regular photograph (like a selfie, a picture of food, a pet, etc.).

        - If the image is NOT an artwork, your response MUST be a valid JSON object with ONLY this structure: {"style": "NOT_AN_ARTWORK", "tags": []}

        - If the image IS an artwork, based on its visual characteristics, please suggest one primary art style and three relevant tags.
        Your response MUST be a valid JSON object with the following structure, and nothing else:
        {
          "style": "SuggestedStyle",
          "tags": ["#tag1", "#tag2", "#tag3"]
        }
        
        Example styles: Realism, Impressionism, Abstract, Pop Art, Surrealism.
        Example tags: #portrait, #landscape, #stilllife, #colorful, #monochrome, #flower.
        """

        response = gemini_model.generate_content([prompt, img])

        cleaned_response = response.text.replace("```json", "").replace("```", "").strip()
        result_json = json.loads(cleaned_response)

        return jsonify(result_json), 200

    except Exception as e:
        print(f"Error during image analysis: {e}")
        return jsonify({"error": "Failed to analyze image"}), 500
