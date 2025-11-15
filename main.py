import os
from flask import Flask, jsonify, request
from supabase import create_client, Client
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import json
import google.generativeai as genai
from PIL import Image
import io

# --- 1. 初始化 Flask App 和 Supabase 客户端 ---
app = Flask(__name__)

# a. Supabase 客户端
SUPABASE_URL = "https://aalhngepjsozacnaeeny.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFhbGhuZ2VwanNvemFjbmFlZW55Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1ODk3MTQ4OSwiZXhwIjoyMDc0NTQ3NDg5fQ.bN6BqyIjFdX4cSRfo02eLGgflctin9pa8zofJVzAeaU" # 注意：这里需要用 Service Role Key，因为它有读写所有表的权限
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# b. Gemini 客户端
GEMINI_API_KEY = "AIzaSyBeLFbY8grT2dAykz-3ht1DQKtdhkzkL_c"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro-latest')


# --- 2. 核心推荐算法 ---
def get_recommendations(user_id: str):
    try:
        # --- a. 获取数据 ---
        likes_data = supabase.table('Like').select('customUserId, artworkId').execute()
        
        # 【冷启动检查 1】如果整个平台都没有任何点赞数据，直接返回
        if not likes_data.data:
            return {"recommended_artwork": None, "recommended_artist": None}

        df = pd.DataFrame(likes_data.data)
        
        # --- b. 创建用户-物品交互矩阵 ---
        interaction_matrix = df.pivot_table(index='customUserId', columns='artworkId', aggfunc='size', fill_value=0)

        # 【冷启动检查 2】如果当前用户不在矩阵里 (新用户)，直接返回
        if user_id not in interaction_matrix.index:
            print(f"User {user_id} is a new user with no likes. No recommendations available.")
            return {"recommended_artwork": None, "recommended_artist": None}

        # --- c. 计算用户相似度 ---
        user_similarity = cosine_similarity(interaction_matrix)
        user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)

        # --- d. 找到最相似的用户 ---
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
        
        # 【冷启动检查 3】如果没有其他用户，或者其他用户都和你不相似，直接返回
        if similar_users.empty or similar_users.iloc[0] == 0:
            return {"recommended_artwork": None, "recommended_artist": None}

        # --- e. 寻找推荐候选作品 ---
        current_user_likes = set(interaction_matrix.loc[user_id][interaction_matrix.loc[user_id] > 0].index)
        
        # 我们不再只看最相似的一个用户，而是遍历所有相似用户，直到找到合适的推荐
        for similar_user_id in similar_users.index:
            similar_user_likes = set(interaction_matrix.loc[similar_user_id][interaction_matrix.loc[similar_user_id] > 0].index)
            
            # 找出相似用户喜欢、但当前用户没喜欢的作品
            recommended_artwork_ids = list(similar_user_likes - current_user_likes)
            
            if not recommended_artwork_ids:
                continue # 如果没有，就看下一个更相似的用户

            # --- f. 获取候选作品的详细信息，并进行过滤 ---
            # 我们一次性获取所有候选作品的信息
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
                    
                    # 【核心修正】查询我们已经创建好的、绝对存在的 'artists_with_details' 视图
                    artist_details = supabase.table('artists_with_details').select('*').eq('customUserId', artwork_owner_id).single().execute()
                    recommended_artist = artist_details.data if artist_details.data else None
                    
                    return {"recommended_artwork": recommended_artwork, "recommended_artist": recommended_artist}
        
        return {"recommended_artwork": None, "recommended_artist": None}

    except Exception as e:
        print(f"CRITICAL Error in recommendation logic for user {user_id}: {e}")
        return {"recommended_artwork": None, "recommended_artist": None}

# --- 3. 图像分析的核心逻辑 ---
def analyze_image_logic(file_stream):
    try:
        # c. 读取图片数据
        img = Image.open(file.stream)

        # d. 定义我们的“提示 (Prompt)”
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

        # e. 把图片和提示一起发送给 Gemini API
        response = model.generate_content([prompt, img])

        # f. 解析 Gemini 返回的 JSON 结果
        # Gemini 的返回结果可能包含一些 Markdown 格式，我们需要清理一下
        cleaned_response = response.text.replace("```json", "").replace("```", "").strip()
        result_json = json.loads(cleaned_response)

        # g. 返回最终结果给 Flutter App
        return jsonify(result_json), 200

    except Exception as e:
        print(f"Error during image analysis: {e}")
        return jsonify({"error": "Failed to analyze image"}), 500


# --- 4. 定义所有的 API 接口 (路由) ---

# 接口 A: 推荐系统
@app.route('/recommend', methods=['GET'])
def recommend_handler():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    recommendations = get_recommendations(user_id)
    return jsonify(recommendations)

# 接口 B: 图像分析
@app.route('/analyze-artwork', methods=['POST'])
def analyze_artwork_handler():
    if 'artwork_image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['artwork_image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        result_json = analyze_image_logic(file.stream)
        return jsonify(result_json), 200
    except Exception as e:
        return jsonify({"error": f"Failed to analyze image: {e}"}), 500


# --- 4. 运行服务器 (仅用于本地测试) ---
if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0') # 我们可以在本地用一个端口测试所有功能
