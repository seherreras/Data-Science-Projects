#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install sentence-transformers


# In[2]:


import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import os
import ssl
from pathlib import Path


# In[3]:


# ===================================================================
# SSL CONFIGURATION FOR RESTRICTED ENVIRONMENTS
# ===================================================================

def configure_ssl_for_local():
    """
    Configure SSL settings to work in restricted environments.
    This allows downloading models when corporate proxies block certificates.
    """
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_CERT_FILE'] = ''
        print("✅ SSL configured for restricted environment")
    except Exception as e:
        print(f"⚠️ Warning configuring SSL: {e}")

# Configure SSL before any imports
configure_ssl_for_local()


# In[4]:


# ===================================================================
# CLASS: OccupationSemanticMatcher (WITH IMPROVED EMBEDDINGS)
# ===================================================================

class OccupationSemanticMatcher:
    """
    Professional system for semantic analysis between occupation tables.
    Uses Bag of Words, TF-IDF, and advanced semantic embeddings.
    """
  
    def __init__(self,
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 use_semantic_embeddings: bool = True,
                 cache_dir: str = './embeddings_cache',
                 local_model_dir: str = './local_models'):
        """
        Args:
            model_name: Sentence-transformers model to use
                       - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual, excellent for Spanish/English
                       - 'all-MiniLM-L6-v2': English, faster
                       - 'all-mpnet-base-v2': English, more accurate but slower
            use_semantic_embeddings: Whether to use semantic embeddings (recommended: True)
            cache_dir: Directory to save embeddings and speed up future runs
            local_model_dir: Directory to save downloaded models locally
        """
        self.model_name = model_name
        self.model = None
        self.use_semantic_embeddings = use_semantic_embeddings
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.local_model_dir = Path(local_model_dir)
        self.local_model_dir.mkdir(exist_ok=True)
        self.setup_logging()
      
    def setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
      
    def load_model(self):
        """Load semantic embeddings model (with local storage support)"""
        if not self.use_semantic_embeddings:
            return
       
        if self.model is not None:
            return
        
        # Path for local model storage
        local_model_path = self.local_model_dir / self.model_name.replace('/', '_')
        
        try:
            # First, try to load from local directory
            if local_model_path.exists():
                self.logger.info(f"🔄 Loading model from local directory: {local_model_path}")
                self.model = SentenceTransformer(str(local_model_path))
                self.logger.info("✅ Model loaded successfully from local storage")
            else:
                # Download and save locally for future use
                self.logger.info(f"🔄 Downloading embeddings model: {self.model_name}")
                self.logger.info("⏱️ This may take a few minutes on first run...")
                
                # Configure SSL before download
                configure_ssl_for_local()
                
                # Download model
                self.model = SentenceTransformer(self.model_name)
                
                # Save model locally for future use
                self.logger.info(f"💾 Saving model locally to: {local_model_path}")
                self.model.save(str(local_model_path))
                self.logger.info("✅ Model downloaded and saved successfully")
                
        except Exception as e:
            self.logger.error(f"❌ Error loading model {self.model_name}: {e}")
            self.logger.info("🔄 Trying fallback model...")
            
            # Try fallback model
            fallback_model = 'all-MiniLM-L6-v2'
            fallback_path = self.local_model_dir / fallback_model.replace('/', '_')
            
            try:
                if fallback_path.exists():
                    self.model = SentenceTransformer(str(fallback_path))
                else:
                    configure_ssl_for_local()
                    self.model = SentenceTransformer(fallback_model)
                    self.model.save(str(fallback_path))
                    
                self.model_name = fallback_model
                self.logger.info("✅ Fallback model loaded successfully")
            except Exception as e2:
                self.logger.error(f"❌ Critical error loading models: {e2}")
                raise
          
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        if pd.isna(text) or text == '':
            return ''
        text = str(text).lower()
        text = re.sub(r'[^\w\s\-/&]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
   
    def get_cache_path(self, texts: List[str], prefix: str) -> Path:
        """Generate cache path based on text hash"""
        text_hash = hash(tuple(texts))
        return self.cache_dir / f"{prefix}_{self.model_name.replace('/', '_')}_{text_hash}.pkl"
   
    def load_embeddings_from_cache(self, cache_path: Path) -> Optional[np.ndarray]:
        """Load embeddings from cache if exists"""
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    embeddings = pickle.load(f)
                self.logger.info(f"✅ Embeddings loaded from cache: {cache_path.name}")
                return embeddings
            except Exception as e:
                self.logger.warning(f"⚠️ Error loading cache: {e}")
        return None
   
    def save_embeddings_to_cache(self, embeddings: np.ndarray, cache_path: Path):
        """Save embeddings to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            self.logger.info(f"💾 Embeddings saved to cache: {cache_path.name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Error saving cache: {e}")
  
    def compute_bow_similarity(self, texts1: List[str], texts2: List[str]) -> Tuple[np.ndarray, CountVectorizer]:
        """Calculate similarity using Bag of Words"""
        all_texts = texts1 + texts2
        vectorizer = CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
        )
        bow_matrix = vectorizer.fit_transform(all_texts)
        bow1 = bow_matrix[:len(texts1)]
        bow2 = bow_matrix[len(texts1):]
        similarity_matrix = cosine_similarity(bow1, bow2)
        return similarity_matrix, vectorizer
  
    def compute_semantic_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """
        Calculate similarity using semantic embeddings (system core).
        Uses cache to speed up repeated runs.
        """
        if self.model is None:
            self.load_model()
       
        # Try loading from cache
        cache_path1 = self.get_cache_path(texts1, "embed1")
        cache_path2 = self.get_cache_path(texts2, "embed2")
       
        embeddings1 = self.load_embeddings_from_cache(cache_path1)
        if embeddings1 is None:
            self.logger.info("🔄 Generating embeddings for table 1...")
            embeddings1 = self.model.encode(
                texts1,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            self.save_embeddings_to_cache(embeddings1, cache_path1)
       
        embeddings2 = self.load_embeddings_from_cache(cache_path2)
        if embeddings2 is None:
            self.logger.info("🔄 Generating embeddings for table 2...")
            embeddings2 = self.model.encode(
                texts2,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            self.save_embeddings_to_cache(embeddings2, cache_path2)
       
        self.logger.info("🔄 Calculating cosine similarity...")
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
       
        return similarity_matrix
  
    def compute_tfidf_similarity(self, texts1: List[str], texts2: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Calculate similarity using TF-IDF"""
        all_texts = texts1 + texts2
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        tfidf1 = tfidf_matrix[:len(texts1)]
        tfidf2 = tfidf_matrix[len(texts1):]
        similarity_matrix = cosine_similarity(tfidf1, tfidf2)
        return similarity_matrix, vectorizer
  
    def find_best_matches(self,
                         df1: pd.DataFrame,
                         df2: pd.DataFrame,
                         text_column1: str,
                         text_column2: str,
                         id_column1: str = None,
                         id_column2: str = None,
                         threshold: float = 0.5,
                         top_n: int = 1,
                         method: str = 'semantic') -> pd.DataFrame:
        """
        Find best matches between two tables.
       
        Args:
            method: 'bow', 'tfidf', 'semantic', 'combined', 'all'
                   - 'semantic': Only embeddings (RECOMMENDED for best accuracy)
                   - 'all': Combines all 3 methods (more robust)
                   - 'combined': Only BoW + TF-IDF
        """
        self.logger.info("📂 Preparing data...")
        df1_clean = df1.copy()
        df2_clean = df2.copy()
      
        if id_column1 is None:
            id_column1 = 'id_table1'
            df1_clean[id_column1] = range(len(df1_clean))
      
        if id_column2 is None:
            id_column2 = 'id_table2'
            df2_clean[id_column2] = range(len(df2_clean))
      
        texts1 = [self.preprocess_text(text) for text in df1_clean[text_column1]]
        texts2 = [self.preprocess_text(text) for text in df2_clean[text_column2]]
       
        bow_sim = None
        tfidf_sim = None
        semantic_sim = None
        vectorizer_bow = None
        vectorizer_tfidf = None
      
        # Calculate similarities based on method
        if method in ['bow', 'combined', 'all']:
            self.logger.info("🔍 Calculating Bag of Words similarities...")
            bow_sim, vectorizer_bow = self.compute_bow_similarity(texts1, texts2)
          
        if method in ['tfidf', 'combined', 'all']:
            self.logger.info("🔍 Calculating TF-IDF similarities...")
            tfidf_sim, vectorizer_tfidf = self.compute_tfidf_similarity(texts1, texts2)
          
        if method in ['semantic', 'all']:
            if not self.use_semantic_embeddings:
                self.logger.warning("⚠️ Semantic embeddings not enabled. Using BoW instead.")
                if bow_sim is None:
                    bow_sim, vectorizer_bow = self.compute_bow_similarity(texts1, texts2)
                similarity_matrix = bow_sim
            else:
                self.logger.info("🧠 Calculating semantic similarities (embeddings)...")
                semantic_sim = self.compute_semantic_similarity(texts1, texts2)
      
        # Combine similarities based on chosen method
        if method == 'bow':
            similarity_matrix = bow_sim
        elif method == 'tfidf':
            similarity_matrix = tfidf_sim
        elif method == 'semantic':
            similarity_matrix = semantic_sim if semantic_sim is not None else bow_sim
        elif method == 'combined':
            similarity_matrix = 0.5 * bow_sim + 0.5 * tfidf_sim
        elif method == 'all':
            if semantic_sim is not None:
                # Give more weight to semantic embeddings (50%), 25% each to BoW and TF-IDF
                similarity_matrix = 0.25 * bow_sim + 0.25 * tfidf_sim + 0.5 * semantic_sim
            else:
                similarity_matrix = 0.5 * bow_sim + 0.5 * tfidf_sim
      
        # Build results
        results = []
      
        for i, (idx1, row1) in enumerate(df1_clean.iterrows()):
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1][:top_n]
          
            for rank, j in enumerate(top_indices):
                similarity_score = similarities[j]
              
                if similarity_score >= threshold:
                    row2 = df2_clean.iloc[j]
                  
                    result = {
                        f'{id_column1}': row1[id_column1],
                        f'occupation_table1': row1[text_column1],
                        f'{id_column2}': row2[id_column2],
                        f'occupation_table2': row2[text_column2],
                        'similarity_percentage': round(similarity_score * 100, 2),
                        'match_ranking': rank + 1,
                        'method_used': method
                    }
                  
                    # Add individual metrics if available
                    if bow_sim is not None:
                        result['similarity_bow'] = round(bow_sim[i, j] * 100, 2)
                    if tfidf_sim is not None:
                        result['similarity_tfidf'] = round(tfidf_sim[i, j] * 100, 2)
                    if semantic_sim is not None:
                        result['similarity_semantic'] = round(semantic_sim[i, j] * 100, 2)
                  
                    results.append(result)
      
        results_df = pd.DataFrame(results)
      
        if len(results_df) > 0:
            results_df = results_df.sort_values('similarity_percentage', ascending=False)
            self.logger.info(f"✅ Process completed. {len(results_df)} matches found.")
        else:
            self.logger.warning("⚠️ No matches found above the specified threshold.")
       
        return results_df
  
    def generate_similarity_report(self, results_df: pd.DataFrame) -> Dict:
        """Generate statistical report of results"""
        if len(results_df) == 0:
            return {"error": "No data to generate report"}
      
        report = {
            "total_matches": len(results_df),
            "average_similarity": round(results_df['similarity_percentage'].mean(), 2),
            "median_similarity": round(results_df['similarity_percentage'].median(), 2),
            "min_similarity": round(results_df['similarity_percentage'].min(), 2),
            "max_similarity": round(results_df['similarity_percentage'].max(), 2),
            "std_similarity": round(results_df['similarity_percentage'].std(), 2),
            "similarity_standard_error": round(results_df['similarity_percentage'].sem(), 2),
            "high_confidence_matches": len(results_df[results_df['similarity_percentage'] >= 80]),
            "medium_confidence_matches": len(results_df[(results_df['similarity_percentage'] >= 60) &
                                                    (results_df['similarity_percentage'] < 80)]),
            "low_confidence_matches": len(results_df[results_df['similarity_percentage'] < 60])
        }
        return report


# In[7]:


# ===================================================================
# MAIN FUNCTION: LOAD AND PROCESS WITH SEMANTIC EMBEDDINGS
# ===================================================================

def load_and_process_tables():
    """
    Main function that loads Excel files and runs analysis with embeddings.
    """
   
    print("=" * 80)
    print(" 🧠 SEMANTIC SIMILARITY ANALYSIS WITH EMBEDDINGS - OCCUPATIONS")
    print("=" * 80)
   
    # ----------------------------------------------------------------
    # STEP 1: LOAD EXCEL FILES
    # ----------------------------------------------------------------
    print("\n📂 STEP 1: Loading Excel files...")
   
    file_path = 'C:/Users/Diego/Downloads/article/NI_Scot.xlsx'
    
    try:
        table1 = pd.read_excel(file_path, sheet_name='Scotland')
        table2 = pd.read_excel(file_path, sheet_name='NorthIre')
       
        print(f" ✅ Table 1 (Scotland): {len(table1)} occupations loaded")
        print(f" ✅ Table 2 (NorthIre): {len(table2)} occupations loaded")
       
    except FileNotFoundError:
        print(f" ❌ ERROR: File not found at {file_path}")
        return None, None, None
    except ValueError as e:
        print(f" ❌ ERROR: {e}")
        xl_file = pd.ExcelFile(file_path)
        print(f" Available sheets: {xl_file.sheet_names}")
        return None, None, None
   
    # ----------------------------------------------------------------
    # STEP 2: CONFIGURE PARAMETERS
    # ----------------------------------------------------------------
    print("\n⚙️ STEP 2: Configuring parameters...")
   
    TEXT_COLUMN_TABLE1 = 'Sub_job_fam'
    TEXT_COLUMN_TABLE2 = 'Job'
    ID_COLUMN_TABLE1 = 'code'
    ID_COLUMN_TABLE2 = 'code'
   
    print(f" 📝 Scotland text column: {TEXT_COLUMN_TABLE1}")
    print(f" 📝 NorthIre text column: {TEXT_COLUMN_TABLE2}")
    print(f" 🎯 Similarity threshold: 30%")
    print(f" 🔢 Top matches per occupation: 3")
    print(f" 🧠 Semantic embeddings: ENABLED")
   
    # ----------------------------------------------------------------
    # STEP 3: INITIALIZE MATCHER WITH EMBEDDINGS
    # ----------------------------------------------------------------
    print("\n🔧 STEP 3: Initializing semantic matcher...")
    matcher = OccupationSemanticMatcher(
        model_name='paraphrase-multilingual-MiniLM-L12-v2',  # Multilingual model
        use_semantic_embeddings=True,  # ✅ EMBEDDINGS ENABLED
        cache_dir='./embeddings_cache',  # Cache to speed up runs
        local_model_dir='./local_models'  # Local model storage
    )
   
    # ----------------------------------------------------------------
    # STEP 4: RUN ANALYSIS WITH EMBEDDINGS
    # ----------------------------------------------------------------
    print("\n🔍 STEP 4: Running similarity analysis...")
    print(" (First run may take longer - generating embeddings)")
   
    results = {}
   
    # Method 1: Pure Semantic Embeddings (RECOMMENDED)
    print("\n → Method 1: Pure Semantic Embeddings 🧠...")
    results['semantic'] = matcher.find_best_matches(
        df1=table1,
        df2=table2,
        text_column1=TEXT_COLUMN_TABLE1,
        text_column2=TEXT_COLUMN_TABLE2,
        id_column1=ID_COLUMN_TABLE1,
        id_column2=ID_COLUMN_TABLE2,
        threshold=0.3,
        top_n=3,
        method='semantic'
    )
   
    # Method 2: Bag of Words (for comparison)
    print("\n → Method 2: Bag of Words (comparison)...")
    results['bow'] = matcher.find_best_matches(
        df1=table1,
        df2=table2,
        text_column1=TEXT_COLUMN_TABLE1,
        text_column2=TEXT_COLUMN_TABLE2,
        id_column1=ID_COLUMN_TABLE1,
        id_column2=ID_COLUMN_TABLE2,
        threshold=0.3,
        top_n=3,
        method='bow'
    )
   
    # Method 3: TF-IDF (for comparison)
    print("\n → Method 3: TF-IDF (comparison)...")
    results['tfidf'] = matcher.find_best_matches(
        df1=table1,
        df2=table2,
        text_column1=TEXT_COLUMN_TABLE1,
        text_column2=TEXT_COLUMN_TABLE2,
        id_column1=ID_COLUMN_TABLE1,
        id_column2=ID_COLUMN_TABLE2,
        threshold=0.3,
        top_n=3,
        method='tfidf'
    )
   
    # Method 4: ALL COMBINED (50% Embeddings + 25% BoW + 25% TF-IDF)
    print("\n → Method 4: Hybrid (50% Embeddings + 25% BoW + 25% TF-IDF) 🚀...")
    results['all'] = matcher.find_best_matches(
        df1=table1,
        df2=table2,
        text_column1=TEXT_COLUMN_TABLE1,
        text_column2=TEXT_COLUMN_TABLE2,
        id_column1=ID_COLUMN_TABLE1,
        id_column2=ID_COLUMN_TABLE2,
        threshold=0.3,
        top_n=3,
        method='all'
    )
   
    # ----------------------------------------------------------------
    # STEP 5: SHOW COMPARATIVE RESULTS
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 📊 COMPARATIVE ANALYSIS RESULTS")
    print("=" * 80)
   
    methods = [
        ('🧠 Semantic Embeddings (RECOMMENDED)', 'semantic'),
        ('📝 Bag of Words', 'bow'),
        ('📊 TF-IDF', 'tfidf'),
        ('🚀 Hybrid (50% Emb + 25% BoW + 25% TF-IDF)', 'all')
    ]
   
    for method_name, method_key in methods:
        df_result = results[method_key]
        print(f"\n{method_name}")
        print("-" * 80)
       
        if len(df_result) > 0:
            report = matcher.generate_similarity_report(df_result)
            print(f" ✅ Matches found: {report['total_matches']}")
            print(f" 📈 Average similarity: {report['average_similarity']:.2f}%")
            print(f" 📊 Median similarity: {report['median_similarity']:.2f}%")
            print(f" 🎯 Maximum similarity: {report['max_similarity']:.2f}%")
            print(f" 📉 Minimum similarity: {report['min_similarity']:.2f}%")
            print(f" 📏 Standard deviation: {report['std_similarity']:.2f}%")
            print(f" ⚡ Standard error: {report['similarity_standard_error']:.2f}%")
            print(f" 🟢 High confidence (>80%): {report['high_confidence_matches']}")
            print(f" 🟡 Medium confidence (60-80%): {report['medium_confidence_matches']}")
            print(f" 🔴 Low confidence (<60%): {report['low_confidence_matches']}")
        else:
            print(" ❌ No matches found")
   
    # ----------------------------------------------------------------
    # STEP 6: SAVE RESULTS
    # ----------------------------------------------------------------
    print("\n💾 STEP 6: Saving results...")
   
    output_path = 'C:/Users/Diego/Downloads/article/match_embed_scot_northi.csv'
   
    for method_key, df_result in results.items():
        if len(df_result) > 0:
            output_file = f'{output_path}match_embed_scot_northi{method_key}.csv'
            df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f" ✅ {method_key.upper()}: {output_file}")
   
    # ----------------------------------------------------------------
    # STEP 7: SHOW TOP MATCH EXAMPLES
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 🏆 TOP 5 BEST MATCHES (Semantic Embeddings Method)")
    print("=" * 80)
   
    df_semantic = results['semantic']
    if len(df_semantic) > 0:
        top_5 = df_semantic.head(5)
        for idx, row in enumerate(top_5.iterrows(), 1):
            _, data = row
            print(f"\n{idx}. SIMILARITY: {data['similarity_percentage']}% {'🟢' if data['similarity_percentage'] >= 80 else '🟡' if data['similarity_percentage'] >= 60 else '🔴'}")
            print(f"   Scotland: {data['occupation_table1']}")
            print(f"   NorthIre:   {data['occupation_table2']}")
            if 'similarity_semantic' in data:
                print(f"   → Semantic: {data['similarity_semantic']}%")
   
    # Compare with hybrid method
    print("\n" + "=" * 80)
    print(" 🚀 TOP 5 BEST MATCHES (Hybrid Method)")
    print("=" * 80)
   
    df_all = results['all']
    if len(df_all) > 0:
        top_5 = df_all.head(5)
        for idx, row in enumerate(top_5.iterrows(), 1):
            _, data = row
            print(f"\n{idx}. SIMILARITY: {data['similarity_percentage']}% {'🟢' if data['similarity_percentage'] >= 80 else '🟡' if data['similarity_percentage'] >= 60 else '🔴'}")
            print(f"   Scotland: {data['occupation_table1']}")
            print(f"   NorthIre:   {data['occupation_table2']}")
            if all(col in data for col in ['similarity_bow', 'similarity_tfidf', 'similarity_semantic']):
                print(f"   → BoW: {data['similarity_bow']}% | TF-IDF: {data['similarity_tfidf']}% | Semantic: {data['similarity_semantic']}%")
   
    print("\n" + "=" * 80)
    print(" ✅ ANALYSIS COMPLETED WITH SEMANTIC EMBEDDINGS")
    print("=" * 80)
    print("\n💡 RECOMMENDATION: Use 'semantic' or 'all' method for best results")
    print("   - 'semantic': Best semantic understanding of context")
    print("   - 'all': More robust by combining multiple methods")
   
    return results, table1, table2

# ===================================================================
# RUN THE SCRIPT WITH ENABLED EMBEDDINGS
# ===================================================================

if __name__ == "__main__":
    print("\n🚀 Starting analysis with semantic embeddings...")
    print("⏱️ Note: First run will take longer (model download)")
    print("💾 Subsequent runs will be much faster (cache)\n")
   
    results, table1, table2 = load_and_process_tables()
   
    if results is not None:
        print("\n📊 Access to results:")
        print(" - results['semantic'] → Pure semantic embeddings")
        print(" - results['bow'] → Bag of Words")
        print(" - results['tfidf'] → TF-IDF")
        print(" - results['all'] → Hybrid method (RECOMMENDED)")


# In[ ]:




