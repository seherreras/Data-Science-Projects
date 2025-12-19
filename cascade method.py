#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
import logging
import time
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from sentence_transformers import SentenceTransformer
import pickle
import os
import ssl
from pathlib import Path
import json
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
print("📥 Checking and downloading required NLTK resources...")

# List of required NLTK resources
required_resources = [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4')
]

for resource_path, resource_name in required_resources:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"   Downloading {resource_name}...")
        try:
            nltk.download(resource_name, quiet=True)
            print(f"   ✅ {resource_name} downloaded")
        except Exception as e:
            print(f"   ⚠️ Could not download {resource_name}: {e}")

print("✅ NLTK resources ready\n")

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

# ===================================================================
# CLASS: OccupationSemanticMatcher (ENHANCED WITH ADVANCED PREPROCESSING)
# ===================================================================

class OccupationSemanticMatcher:
    """
    Professional system for semantic analysis between occupation tables.
    Uses Bag of Words, TF-IDF, and advanced semantic embeddings.
    Enhanced with detailed reporting, ablation analysis, and advanced preprocessing.
    
    New Features:
    - Tokenization using NLTK word_tokenize
    - Stopwords removal (multi-language support)
    - Lemmatization using WordNet Lemmatizer
    - Comprehensive preprocessing statistics tracking
    - Configurable preprocessing pipeline
    """
  
    def __init__(self,
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 use_semantic_embeddings: bool = True,
                 cache_dir: str = './embeddings_cache',
                 local_model_dir: str = './local_models',
                 batch_size: int = 32,
                 seed: int = 42,
                 use_advanced_preprocessing: bool = True,
                 language: str = 'english'):
        """
        Args:
            model_name: Sentence-transformers model to use
            use_semantic_embeddings: Whether to use semantic embeddings
            cache_dir: Directory to save embeddings
            local_model_dir: Directory to save downloaded models
            batch_size: Batch size for encoding (affects performance)
            seed: Random seed for reproducibility
            use_advanced_preprocessing: Enable tokenization, stopwords, lemmatization
            language: Language for stopwords and tokenization (english, spanish, french, etc.)
        """
        self.model_name = model_name
        self.model = None
        self.use_semantic_embeddings = use_semantic_embeddings
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.local_model_dir = Path(local_model_dir)
        self.local_model_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.seed = seed
        self.use_advanced_preprocessing = use_advanced_preprocessing
        self.language = language
        
        # Initialize NLTK components if advanced preprocessing is enabled
        if self.use_advanced_preprocessing:
            self.lemmatizer = WordNetLemmatizer()
            try:
                self.stop_words = set(stopwords.words(language))
            except:
                print(f"⚠️ Stopwords for '{language}' not available, using English")
                self.stop_words = set(stopwords.words('english'))
        else:
            self.lemmatizer = None
            self.stop_words = set()
        
        # Preprocessing statistics
        self.preprocessing_stats = {
            'total_texts_processed': 0,
            'total_tokens_before': 0,
            'total_tokens_after': 0,
            'stopwords_removed': 0,
            'tokens_lemmatized': 0,
            'avg_tokens_per_text_before': 0,
            'avg_tokens_per_text_after': 0,
            'preprocessing_method': 'advanced' if use_advanced_preprocessing else 'basic'
        }
        
        # Runtime tracking
        self.runtime_stats = {
            'model_loading_time': 0,
            'embedding_time': 0,
            'similarity_computation_time': 0,
            'total_time': 0,
            'preprocessing_time': 0
        }
        
        # Model metadata
        self.model_metadata = {}
        
        self.setup_logging()
        np.random.seed(seed)
      
    def setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
      
    def load_model(self):
        """Load semantic embeddings model with metadata extraction and corruption handling"""
        if not self.use_semantic_embeddings:
            return
       
        if self.model is not None:
            return
        
        start_time = time.time()
        
        # Path for local model storage
        local_model_path = self.local_model_dir / self.model_name.replace('/', '_')
        
        try:
            # First, try to load from local directory
            if local_model_path.exists():
                self.logger.info(f"📂 Loading model from local directory: {local_model_path}")
                try:
                    self.model = SentenceTransformer(str(local_model_path))
                    self.logger.info("✅ Model loaded successfully from local storage")
                except Exception as load_error:
                    self.logger.warning(f"⚠️ Corrupted local model detected: {load_error}")
                    self.logger.info("🗑️ Deleting corrupted model cache...")
                    
                    # Delete corrupted model directory
                    import shutil
                    try:
                        shutil.rmtree(local_model_path)
                        self.logger.info("✅ Corrupted cache deleted")
                    except Exception as del_error:
                        self.logger.warning(f"⚠️ Could not delete cache: {del_error}")
                    
                    # Force re-download
                    self.logger.info(f"📥 Re-downloading model: {self.model_name}")
                    configure_ssl_for_local()
                    self.model = SentenceTransformer(self.model_name)
                    self.model.save(str(local_model_path))
                    self.logger.info("✅ Model re-downloaded and saved successfully")
            else:
                # Download and save locally for future use
                self.logger.info(f"📥 Downloading embeddings model: {self.model_name}")
                self.logger.info("⏱️ This may take a few minutes on first run...")
                
                configure_ssl_for_local()
                self.model = SentenceTransformer(self.model_name)
                
                # Save model locally for future use
                self.logger.info(f"💾 Saving model locally to: {local_model_path}")
                self.model.save(str(local_model_path))
                self.logger.info("✅ Model downloaded and saved successfully")
            
            # Extract model metadata
            self._extract_model_metadata()
                
        except Exception as e:
            self.logger.error(f"❌ Error loading model {self.model_name}: {e}")
            self.logger.info("📥 Trying fallback model...")
            
            # Try fallback model
            fallback_model = 'all-MiniLM-L6-v2'
            fallback_path = self.local_model_dir / fallback_model.replace('/', '_')
            
            try:
                if fallback_path.exists():
                    try:
                        self.model = SentenceTransformer(str(fallback_path))
                    except Exception as fb_error:
                        self.logger.warning(f"⚠️ Fallback model also corrupted: {fb_error}")
                        import shutil
                        shutil.rmtree(fallback_path)
                        configure_ssl_for_local()
                        self.model = SentenceTransformer(fallback_model)
                        self.model.save(str(fallback_path))
                else:
                    configure_ssl_for_local()
                    self.model = SentenceTransformer(fallback_model)
                    self.model.save(str(fallback_path))
                    
                self.model_name = fallback_model
                self._extract_model_metadata()
                self.logger.info("✅ Fallback model loaded successfully")
            except Exception as e2:
                self.logger.error(f"❌ Critical error loading models: {e2}")
                raise
        
        self.runtime_stats['model_loading_time'] = time.time() - start_time
    
    def _extract_model_metadata(self):
        """Extract detailed metadata from the loaded model"""
        if self.model is None:
            return
        
        try:
            # Get model configuration
            pooling_config = None
            max_seq_length = None
            
            # Try to get pooling configuration
            if hasattr(self.model, '_modules'):
                for module in self.model._modules.values():
                    if hasattr(module, 'pooling_mode'):
                        pooling_config = {
                            'pooling_mode_cls_token': getattr(module, 'pooling_mode_cls_token', False),
                            'pooling_mode_mean_tokens': getattr(module, 'pooling_mode_mean_tokens', False),
                            'pooling_mode_max_tokens': getattr(module, 'pooling_mode_max_tokens', False),
                            'pooling_mode_mean_sqrt_len_tokens': getattr(module, 'pooling_mode_mean_sqrt_len_tokens', False)
                        }
                        break
            
            # Get max sequence length
            if hasattr(self.model, 'max_seq_length'):
                max_seq_length = self.model.max_seq_length
            elif hasattr(self.model, '_first_module') and hasattr(self.model._first_module(), 'max_seq_length'):
                max_seq_length = self.model._first_module().max_seq_length
            
            # Get embedding dimension
            embedding_dim = self.model.get_sentence_embedding_dimension()
            
            self.model_metadata = {
                'model_variant': self.model_name,
                'pooling_strategy': self._get_pooling_mode_name(pooling_config) if pooling_config else 'Unknown',
                'max_sequence_length': max_seq_length if max_seq_length else 'Unknown',
                'embedding_dimension': embedding_dim,
                'batch_size': self.batch_size,
                'random_seed': self.seed,
                'pooling_config_details': pooling_config
            }
            
            self.logger.info(f"📊 Model metadata extracted: {self.model_metadata['model_variant']}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract full model metadata: {e}")
            self.model_metadata = {
                'model_variant': self.model_name,
                'pooling_strategy': 'Unknown',
                'max_sequence_length': 'Unknown',
                'embedding_dimension': 'Unknown',
                'batch_size': self.batch_size,
                'random_seed': self.seed
            }
    
    def _get_pooling_mode_name(self, pooling_config):
        """Convert pooling configuration to readable name"""
        if pooling_config is None:
            return 'Unknown'
        
        if pooling_config.get('pooling_mode_mean_tokens'):
            return 'Mean pooling'
        elif pooling_config.get('pooling_mode_cls_token'):
            return 'CLS token'
        elif pooling_config.get('pooling_mode_max_tokens'):
            return 'Max pooling'
        elif pooling_config.get('pooling_mode_mean_sqrt_len_tokens'):
            return 'Mean sqrt pooling'
        else:
            return 'Custom pooling'
    
    def get_model_report(self) -> Dict:
        """Generate comprehensive model report including preprocessing details"""
        report = {
            'model_configuration': self.model_metadata,
            'preprocessing_configuration': {
                'method': self.preprocessing_stats['preprocessing_method'],
                'language': self.language,
                'tokenization_enabled': self.use_advanced_preprocessing,
                'stopwords_removal_enabled': self.use_advanced_preprocessing,
                'lemmatization_enabled': self.use_advanced_preprocessing,
                'stopwords_count': len(self.stop_words) if self.use_advanced_preprocessing else 0
            },
            'preprocessing_statistics': self.preprocessing_stats,
            'runtime_statistics': self.runtime_stats,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return report
      
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text with detailed tracking of tokenization, stopwords, and lemmatization.
        
        Advanced preprocessing pipeline:
        1. Lowercase conversion
        2. Special character cleaning
        3. Tokenization (word_tokenize)
        4. Stopwords removal
        5. Lemmatization (WordNet)
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        if self.use_advanced_preprocessing:
            # Track statistics
            original_text = text
            
            # Step 1: Clean special characters but keep word boundaries
            text = re.sub(r'[^\w\s\-/&]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Step 2: Tokenization
            tokens = word_tokenize(text)
            tokens_before = len(tokens)
            
            # Step 3: Remove stopwords
            tokens_without_stopwords = [token for token in tokens if token not in self.stop_words]
            stopwords_removed = tokens_before - len(tokens_without_stopwords)
            
            # Step 4: Lemmatization (only on alphabetic tokens of length > 2)
            lemmatized_tokens = []
            for token in tokens_without_stopwords:
                if token.isalpha() and len(token) > 2:
                    lemmatized_tokens.append(self.lemmatizer.lemmatize(token))
                elif len(token) > 0:
                    lemmatized_tokens.append(token)
            
            tokens_after = len(lemmatized_tokens)
            
            # Update statistics
            self.preprocessing_stats['total_texts_processed'] += 1
            self.preprocessing_stats['total_tokens_before'] += tokens_before
            self.preprocessing_stats['total_tokens_after'] += tokens_after
            self.preprocessing_stats['stopwords_removed'] += stopwords_removed
            self.preprocessing_stats['tokens_lemmatized'] += tokens_after
            
            # Reconstruct text
            processed_text = ' '.join(lemmatized_tokens)
            
            return processed_text
        else:
            # Basic preprocessing (original method)
            text = re.sub(r'[^\w\s\-/&]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    
    def finalize_preprocessing_stats(self):
        """Calculate average statistics after all preprocessing is done"""
        if self.preprocessing_stats['total_texts_processed'] > 0:
            self.preprocessing_stats['avg_tokens_per_text_before'] = round(
                self.preprocessing_stats['total_tokens_before'] / 
                self.preprocessing_stats['total_texts_processed'], 2
            )
            self.preprocessing_stats['avg_tokens_per_text_after'] = round(
                self.preprocessing_stats['total_tokens_after'] / 
                self.preprocessing_stats['total_texts_processed'], 2
            )
            self.preprocessing_stats['stopwords_removal_rate'] = round(
                (self.preprocessing_stats['stopwords_removed'] / 
                 self.preprocessing_stats['total_tokens_before'] * 100) if 
                self.preprocessing_stats['total_tokens_before'] > 0 else 0, 2
            )
            self.preprocessing_stats['token_reduction_rate'] = round(
                ((self.preprocessing_stats['total_tokens_before'] - 
                  self.preprocessing_stats['total_tokens_after']) / 
                 self.preprocessing_stats['total_tokens_before'] * 100) if 
                self.preprocessing_stats['total_tokens_before'] > 0 else 0, 2
            )
   
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
        start_time = time.time()
        
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
        
        self.runtime_stats['bow_computation_time'] = time.time() - start_time
        return similarity_matrix, vectorizer
  
    def compute_semantic_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """
        Calculate similarity using semantic embeddings with runtime tracking.
        """
        if self.model is None:
            self.load_model()
       
        embed_start_time = time.time()
        
        # Try loading from cache
        cache_path1 = self.get_cache_path(texts1, "embed1")
        cache_path2 = self.get_cache_path(texts2, "embed2")
       
        embeddings1 = self.load_embeddings_from_cache(cache_path1)
        if embeddings1 is None:
            self.logger.info("📥 Generating embeddings for table 1...")
            embeddings1 = self.model.encode(
                texts1,
                show_progress_bar=True,
                batch_size=self.batch_size,
                convert_to_numpy=True
            )
            self.save_embeddings_to_cache(embeddings1, cache_path1)
       
        embeddings2 = self.load_embeddings_from_cache(cache_path2)
        if embeddings2 is None:
            self.logger.info("📥 Generating embeddings for table 2...")
            embeddings2 = self.model.encode(
                texts2,
                show_progress_bar=True,
                batch_size=self.batch_size,
                convert_to_numpy=True
            )
            self.save_embeddings_to_cache(embeddings2, cache_path2)
        
        self.runtime_stats['embedding_time'] = time.time() - embed_start_time
       
        sim_start_time = time.time()
        self.logger.info("📊 Calculating cosine similarity...")
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        self.runtime_stats['similarity_computation_time'] = time.time() - sim_start_time
       
        return similarity_matrix
  
    def compute_tfidf_similarity(self, texts1: List[str], texts2: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Calculate similarity using TF-IDF"""
        start_time = time.time()
        
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
        
        self.runtime_stats['tfidf_computation_time'] = time.time() - start_time
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
                         method: str = 'semantic',
                         ensemble_weights: Tuple[float, float, float] = (0.25, 0.25, 0.5)) -> pd.DataFrame:
        """
        Find best matches between two tables.
       
        Args:
            method: 'bow', 'tfidf', 'semantic', 'combined', 'all'
            ensemble_weights: (bow_weight, tfidf_weight, semantic_weight) for 'all' method
        """
        total_start_time = time.time()
        
        self.logger.info("📂 Preparing data...")
        df1_clean = df1.copy()
        df2_clean = df2.copy()
      
        if id_column1 is None:
            id_column1 = 'id_table1'
            df1_clean[id_column1] = range(len(df1_clean))
      
        if id_column2 is None:
            id_column2 = 'id_table2'
            df2_clean[id_column2] = range(len(df2_clean))
      
        # Preprocessing with timing
        preprocess_start = time.time()
        self.logger.info(f"🔧 Preprocessing texts (method: {self.preprocessing_stats['preprocessing_method']})...")
        
        texts1 = [self.preprocess_text(text) for text in df1_clean[text_column1]]
        texts2 = [self.preprocess_text(text) for text in df2_clean[text_column2]]
        
        # Finalize preprocessing statistics
        self.finalize_preprocessing_stats()
        
        self.runtime_stats['preprocessing_time'] = time.time() - preprocess_start
        
        # Log preprocessing results
        if self.use_advanced_preprocessing:
            self.logger.info(f"✅ Preprocessing completed:")
            self.logger.info(f"   📊 Texts processed: {self.preprocessing_stats['total_texts_processed']}")
            self.logger.info(f"   📝 Avg tokens before: {self.preprocessing_stats['avg_tokens_per_text_before']}")
            self.logger.info(f"   📝 Avg tokens after: {self.preprocessing_stats['avg_tokens_per_text_after']}")
            self.logger.info(f"   🗑️ Stopwords removed: {self.preprocessing_stats['stopwords_removed']} ({self.preprocessing_stats.get('stopwords_removal_rate', 0)}%)")
            self.logger.info(f"   🔄 Token reduction: {self.preprocessing_stats.get('token_reduction_rate', 0)}%")
       
        bow_sim = None
        tfidf_sim = None
        semantic_sim = None
        vectorizer_bow = None
        vectorizer_tfidf = None
      
        # Calculate similarities based on method
        if method in ['bow', 'combined', 'all']:
            self.logger.info("📊 Calculating Bag of Words similarities...")
            bow_sim, vectorizer_bow = self.compute_bow_similarity(texts1, texts2)
          
        if method in ['tfidf', 'combined', 'all']:
            self.logger.info("📊 Calculating TF-IDF similarities...")
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
            bow_weight, tfidf_weight, semantic_weight = ensemble_weights
            if semantic_sim is not None:
                similarity_matrix = (bow_weight * bow_sim + 
                                   tfidf_weight * tfidf_sim + 
                                   semantic_weight * semantic_sim)
            else:
                # Normalize weights if semantic is not available
                total_weight = bow_weight + tfidf_weight
                similarity_matrix = ((bow_weight/total_weight) * bow_sim + 
                                   (tfidf_weight/total_weight) * tfidf_sim)
      
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
                        'method_used': method,
                        'threshold_used': threshold
                    }
                  
                    # Add ensemble weights if using 'all' method
                    if method == 'all':
                        result['ensemble_weights'] = f"BoW:{bow_weight:.2f}|TF-IDF:{tfidf_weight:.2f}|Semantic:{semantic_weight:.2f}"
                  
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
        
        self.runtime_stats['total_time'] = time.time() - total_start_time
       
        return results_df
  
    def generate_similarity_report(self, results_df: pd.DataFrame) -> Dict:
        """Generate statistical report of results"""
        if len(results_df) == 0:
            return {"error": "No data to generate report"}
      
        report = {
            "total_matches": int(len(results_df)),
            "average_similarity": float(round(results_df['similarity_percentage'].mean(), 2)),
            "median_similarity": float(round(results_df['similarity_percentage'].median(), 2)),
            "min_similarity": float(round(results_df['similarity_percentage'].min(), 2)),
            "max_similarity": float(round(results_df['similarity_percentage'].max(), 2)),
            "std_similarity": float(round(results_df['similarity_percentage'].std(), 2)),
            "similarity_standard_error": float(round(results_df['similarity_percentage'].sem(), 2)),
            "high_confidence_matches": int(len(results_df[results_df['similarity_percentage'] >= 80])),
            "medium_confidence_matches": int(len(results_df[(results_df['similarity_percentage'] >= 60) &
                                                (results_df['similarity_percentage'] < 80)])),
            "low_confidence_matches": int(len(results_df[results_df['similarity_percentage'] < 60]))
        }
        return report


# ===================================================================
# THRESHOLD SENSITIVITY ANALYSIS
# ===================================================================

def threshold_sensitivity_analysis(matcher, df1, df2, text_col1, text_col2, 
                                   id_col1, id_col2, method='semantic'):
    """
    Analyze sensitivity to threshold values (30%, 40%, 50%).
    """
    print("\n" + "=" * 80)
    print(" 🔍 THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    thresholds = [0.30, 0.40, 0.50]
    sensitivity_results = {}
    
    for threshold in thresholds:
        print(f"\n📊 Testing threshold: {threshold*100:.0f}%")
        
        results = matcher.find_best_matches(
            df1=df1,
            df2=df2,
            text_column1=text_col1,
            text_column2=text_col2,
            id_column1=id_col1,
            id_column2=id_col2,
            threshold=threshold,
            top_n=3,
            method=method
        )
        
        if len(results) > 0:
            report = matcher.generate_similarity_report(results)
            sensitivity_results[f"{threshold*100:.0f}%"] = {
                'threshold': threshold,
                'total_matches': report['total_matches'],
                'avg_similarity': report['average_similarity'],
                'median_similarity': report['median_similarity'],
                'high_confidence': report['high_confidence_matches'],
                'medium_confidence': report['medium_confidence_matches'],
                'low_confidence': report['low_confidence_matches'],
                'results_df': results
            }
            
            print(f"  ✅ Matches: {report['total_matches']}")
            print(f"  📈 Avg similarity: {report['average_similarity']:.2f}%")
            print(f"  📊 Median: {report['median_similarity']:.2f}%")
            print(f"  🟢 High conf (>80%): {report['high_confidence_matches']}")
            print(f"  🟡 Med conf (60-80%): {report['medium_confidence_matches']}")
            print(f"  🔴 Low conf (<60%): {report['low_confidence_matches']}")
        else:
            print(f"  ❌ No matches found")
            sensitivity_results[f"{threshold*100:.0f}%"] = {
                'threshold': threshold,
                'total_matches': 0
            }
    
    # Summary comparison
    print("\n" + "-" * 80)
    print(" 📊 THRESHOLD COMPARISON SUMMARY")
    print("-" * 80)
    print(f"{'Threshold':<12} {'Matches':<10} {'Avg Sim':<12} {'High Conf':<12}")
    print("-" * 80)
    
    for thresh_name, data in sensitivity_results.items():
        if data['total_matches'] > 0:
            print(f"{thresh_name:<12} {data['total_matches']:<10} "
                  f"{data['avg_similarity']:<12.2f} {data['high_confidence']:<12}")
        else:
            print(f"{thresh_name:<12} {data['total_matches']:<10} {'N/A':<12} {'N/A':<12}")
    
    return sensitivity_results


# ===================================================================
# ENSEMBLE WEIGHTS ABLATION ANALYSIS
# ===================================================================

def ensemble_ablation_analysis(matcher, df1, df2, text_col1, text_col2, 
                               id_col1, id_col2, threshold=0.3):
    """
    Ablation study on ensemble weight combinations.
    """
    print("\n" + "=" * 80)
    print(" 🔬 ENSEMBLE WEIGHTS ABLATION ANALYSIS")
    print("=" * 80)
    
    # Different weight configurations (bow, tfidf, semantic)
    weight_configs = [
        (0.33, 0.33, 0.34, "Equal (33/33/34)"),
        (0.25, 0.25, 0.50, "Default (25/25/50)"),
        (0.20, 0.20, 0.60, "Semantic-heavy (20/20/60)"),
        (0.15, 0.15, 0.70, "More semantic (15/15/70)"),
        (0.10, 0.10, 0.80, "Very semantic (10/10/80)"),
        (0.40, 0.40, 0.20, "Traditional-heavy (40/40/20)"),
        (0.50, 0.25, 0.25, "BoW-heavy (50/25/25)"),
        (0.25, 0.50, 0.25, "TF-IDF-heavy (25/50/25)")
    ]
    
    ablation_results = {}
    
    for bow_w, tfidf_w, sem_w, config_name in weight_configs:
        print(f"\n🔬 Testing: {config_name}")
        print(f"   Weights → BoW:{bow_w:.2f} | TF-IDF:{tfidf_w:.2f} | Semantic:{sem_w:.2f}")
        
        results = matcher.find_best_matches(
            df1=df1,
            df2=df2,
            text_column1=text_col1,
            text_column2=text_col2,
            id_column1=id_col1,
            id_column2=id_col2,
            threshold=threshold,
            top_n=3,
            method='all',
            ensemble_weights=(bow_w, tfidf_w, sem_w)
        )
        
        if len(results) > 0:
            report = matcher.generate_similarity_report(results)
            ablation_results[config_name] = {
                'weights': (bow_w, tfidf_w, sem_w),
                'total_matches': report['total_matches'],
                'avg_similarity': report['average_similarity'],
                'median_similarity': report['median_similarity'],
                'std_similarity': report['std_similarity'],
                'high_confidence': report['high_confidence_matches'],
                'medium_confidence': report['medium_confidence_matches'],
                'low_confidence': report['low_confidence_matches'],
                'results_df': results
            }
            
            print(f"   ✅ Matches: {report['total_matches']}")
            print(f"   📈 Avg: {report['average_similarity']:.2f}%")
            print(f"   📊 Median: {report['median_similarity']:.2f}%")
            print(f"   📉 Std Dev: {report['std_similarity']:.2f}%")
            print(f"   🟢 High: {report['high_confidence_matches']} | "
                  f"🟡 Med: {report['medium_confidence_matches']} | "
                  f"🔴 Low: {report['low_confidence_matches']}")
        else:
            ablation_results[config_name] = {
                'weights': (bow_w, tfidf_w, sem_w),
                'total_matches': 0
            }
            print(f"   ❌ No matches found")
    
    # Summary table
    print("\n" + "-" * 100)
    print(" 📊 ABLATION SUMMARY TABLE")
    print("-" * 100)
    print(f"{'Configuration':<25} {'Weights (B/T/S)':<20} {'Matches':<10} "
          f"{'Avg Sim':<10} {'High':<8}")
    print("-" * 100)
    
    for config_name, data in ablation_results.items():
        if data['total_matches'] > 0:
            w = data['weights']
            print(f"{config_name:<25} {f'{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}':<20} "
                  f"{data['total_matches']:<10} {data['avg_similarity']:<10.2f} "
                  f"{data['high_confidence']:<8}")
        else:
            w = data['weights']
            print(f"{config_name:<25} {f'{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}':<20} "
                  f"{data['total_matches']:<10} {'N/A':<10} {'N/A':<8}")
    
    # Find best configuration
    best_config = max(
        [(name, data) for name, data in ablation_results.items() if data['total_matches'] > 0],
        key=lambda x: x[1]['avg_similarity'],
        default=(None, None)
    )
    
    if best_config[0]:
        print("\n" + "=" * 100)
        print(f" 🏆 BEST CONFIGURATION: {best_config[0]}")
        print(f" 📈 Average Similarity: {best_config[1]['avg_similarity']:.2f}%")
        print(f" 🎯 Total Matches: {best_config[1]['total_matches']}")
        print("=" * 100)
    
    return ablation_results


# ===================================================================
# HELPER FUNCTION: CONVERT TO JSON SERIALIZABLE
# ===================================================================

def convert_to_json_serializable(obj):
    """
    Convert NumPy types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# ===================================================================
# GENERATE COMPREHENSIVE REPORT
# ===================================================================

def generate_comprehensive_report(matcher, sensitivity_results, ablation_results, 
                                  output_dir='./reports'):
    """
    Generate comprehensive JSON and text reports.
    """
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Model report
    model_report = matcher.get_model_report()
    
    # Compile full report (excluding DataFrames)
    full_report = {
        'timestamp': timestamp,
        'model_configuration': model_report['model_configuration'],
        'preprocessing_configuration': model_report['preprocessing_configuration'],
        'preprocessing_statistics': model_report['preprocessing_statistics'],
        'runtime_statistics': model_report['runtime_statistics'],
        'threshold_sensitivity': {
            name: {k: v for k, v in data.items() if k != 'results_df'}
            for name, data in sensitivity_results.items()
        },
        'ensemble_ablation': {
            name: {k: v for k, v in data.items() if k != 'results_df'}
            for name, data in ablation_results.items()
        }
    }
    
    # Convert all NumPy types to native Python types
    full_report = convert_to_json_serializable(full_report)
    
    # Save JSON report
    json_path = f"{output_dir}/comprehensive_report_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 JSON Report saved: {json_path}")
    
    # Save text report
    txt_path = f"{output_dir}/comprehensive_report_{timestamp}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(" COMPREHENSIVE SEMANTIC MATCHING ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("1. MODEL CONFIGURATION\n")
        f.write("-" * 100 + "\n")
        for key, value in model_report['model_configuration'].items():
            if key != 'pooling_config_details':
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("2. RUNTIME STATISTICS\n")
        f.write("-" * 100 + "\n")
        for key, value in model_report['runtime_statistics'].items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f} seconds\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("3. PREPROCESSING CONFIGURATION\n")
        f.write("-" * 100 + "\n")
        preproc_config = model_report.get('preprocessing_configuration', {})
        for key, value in preproc_config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("4. PREPROCESSING STATISTICS\n")
        f.write("-" * 100 + "\n")
        preproc_stats = model_report.get('preprocessing_statistics', {})
        for key, value in preproc_stats.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("5. THRESHOLD SENSITIVITY ANALYSIS\n")
        f.write("-" * 100 + "\n")
        for name, data in sensitivity_results.items():
            f.write(f"\nThreshold: {name}\n")
            if data['total_matches'] > 0:
                f.write(f"  Total matches: {data['total_matches']}\n")
                f.write(f"  Average similarity: {data['avg_similarity']:.2f}%\n")
                f.write(f"  High confidence: {data['high_confidence']}\n")
                f.write(f"  Medium confidence: {data['medium_confidence']}\n")
                f.write(f"  Low confidence: {data['low_confidence']}\n")
            else:
                f.write(f"  No matches found\n")
        f.write("\n")
        
        f.write("6. ENSEMBLE WEIGHTS ABLATION STUDY\n")
        f.write("-" * 100 + "\n")
        for name, data in ablation_results.items():
            f.write(f"\n{name}\n")
            w = data['weights']
            f.write(f"  Weights (BoW/TF-IDF/Semantic): {w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}\n")
            if data['total_matches'] > 0:
                f.write(f"  Total matches: {data['total_matches']}\n")
                f.write(f"  Average similarity: {data['avg_similarity']:.2f}%\n")
                f.write(f"  Std deviation: {data['std_similarity']:.2f}%\n")
            else:
                f.write(f"  No matches found\n")
    
    print(f"💾 Text Report saved: {txt_path}")
    
    return full_report


# ===================================================================
# MANUAL VALIDATION SYSTEM
# ===================================================================

def create_validation_sample(results_df: pd.DataFrame, sample_size: int = 100, 
                            stratified: bool = True, random_state: int = 42) -> pd.DataFrame:
    """
    Create a stratified sample for manual validation.
    
    Args:
        results_df: DataFrame with matching results
        sample_size: Number of samples to validate
        stratified: Whether to stratify by confidence levels
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with validation sample
    """
    print("\n" + "=" * 80)
    print(" 📝 CREATING VALIDATION SAMPLE")
    print("=" * 80)
    
    if len(results_df) == 0:
        print("❌ No results to sample from")
        return pd.DataFrame()
    
    # Add confidence categories
    def get_confidence_category(sim):
        if sim >= 80:
            return 'high'
        elif sim >= 60:
            return 'medium'
        else:
            return 'low'
    
    results_df['confidence_category'] = results_df['similarity_percentage'].apply(get_confidence_category)
    
    if stratified and len(results_df) >= sample_size:
        # Stratified sampling by confidence level
        n_high = min(int(sample_size * 0.33), len(results_df[results_df['confidence_category'] == 'high']))
        n_medium = min(int(sample_size * 0.33), len(results_df[results_df['confidence_category'] == 'medium']))
        n_low = sample_size - n_high - n_medium
        
        high_sample = results_df[results_df['confidence_category'] == 'high'].sample(
            n=n_high, random_state=random_state) if n_high > 0 else pd.DataFrame()
        medium_sample = results_df[results_df['confidence_category'] == 'medium'].sample(
            n=n_medium, random_state=random_state) if n_medium > 0 else pd.DataFrame()
        low_sample = results_df[results_df['confidence_category'] == 'low'].sample(
            n=n_low, random_state=random_state) if n_low > 0 else pd.DataFrame()
        
        sample = pd.concat([high_sample, medium_sample, low_sample], ignore_index=True)
        
        print(f"✅ Stratified sample created:")
        print(f"   🟢 High confidence: {len(high_sample)}")
        print(f"   🟡 Medium confidence: {len(medium_sample)}")
        print(f"   🔴 Low confidence: {len(low_sample)}")
    else:
        # Simple random sampling
        sample_size = min(sample_size, len(results_df))
        sample = results_df.sample(n=sample_size, random_state=random_state)
        print(f"✅ Random sample created: {len(sample)} items")
    
    # Add validation columns
    sample['annotator_1'] = ''
    sample['annotator_2'] = ''
    sample['gold_standard'] = ''
    sample['notes'] = ''
    
    return sample.reset_index(drop=True)


def load_annotations(validation_file: str) -> pd.DataFrame:
    """
    Load annotated validation file.
    
    Expected columns:
    - All original columns from the sample
    - annotator_1: 1 (correct match), 0 (incorrect match)
    - annotator_2: 1 (correct match), 0 (incorrect match)
    - gold_standard: Final agreed label (optional)
    """
    print("\n📂 Loading annotations...")
    
    try:
        annotations = pd.read_csv(validation_file, encoding='utf-8-sig')
        print(f"✅ Loaded {len(annotations)} annotated samples")
        return annotations
    except FileNotFoundError:
        print(f"❌ File not found: {validation_file}")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def calculate_inter_annotator_agreement(annotations: pd.DataFrame) -> Dict:
    """
    Calculate inter-annotator agreement metrics.
    
    Returns:
        Dictionary with agreement metrics
    """
    print("\n" + "=" * 80)
    print(" 🤝 INTER-ANNOTATOR AGREEMENT ANALYSIS")
    print("=" * 80)
    
    # Convert to numeric if needed
    annotations['annotator_1'] = pd.to_numeric(annotations['annotator_1'], errors='coerce')
    annotations['annotator_2'] = pd.to_numeric(annotations['annotator_2'], errors='coerce')
    
    # Remove rows with missing annotations
    valid_annotations = annotations.dropna(subset=['annotator_1', 'annotator_2'])
    
    if len(valid_annotations) == 0:
        print("❌ No valid annotations found")
        return {}
    
    annotator_1 = valid_annotations['annotator_1'].values
    annotator_2 = valid_annotations['annotator_2'].values
    
    # Calculate agreement rate (percentage agreement)
    agreement = (annotator_1 == annotator_2).sum()
    total = len(annotator_1)
    agreement_rate = (agreement / total) * 100
    
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(annotator_1, annotator_2)
    
    # Disagreement analysis
    disagreements = valid_annotations[annotator_1 != annotator_2]
    
    results = {
        'total_annotations': total,
        'agreements': int(agreement),
        'disagreements': int(total - agreement),
        'agreement_rate': round(agreement_rate, 2),
        'cohens_kappa': round(kappa, 4),
        'kappa_interpretation': interpret_kappa(kappa),
        'disagreement_cases': disagreements
    }
    
    # Print results
    print(f"\n📊 Agreement Statistics:")
    print(f"   Total annotations: {results['total_annotations']}")
    print(f"   Agreements: {results['agreements']}")
    print(f"   Disagreements: {results['disagreements']}")
    print(f"   Agreement rate: {results['agreement_rate']:.2f}%")
    print(f"   Cohen's Kappa: {results['cohens_kappa']:.4f} ({results['kappa_interpretation']})")
    
    if len(disagreements) > 0:
        print(f"\n⚠️ Disagreement cases: {len(disagreements)}")
        print(f"   (See detailed report for full list)")
    
    return results


def interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's Kappa value"""
    if kappa < 0:
        return "Poor (Less than chance agreement)"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def calculate_validation_metrics(annotations: pd.DataFrame, method_name: str = 'Model') -> Dict:
    """
    Calculate precision, recall, and F1 score based on gold standard labels.
    
    Args:
        annotations: DataFrame with 'gold_standard' column and model predictions
        method_name: Name of the method being evaluated
    
    Returns:
        Dictionary with validation metrics
    """
    print("\n" + "=" * 80)
    print(f" 📈 VALIDATION METRICS - {method_name}")
    print("=" * 80)
    
    # Use gold standard if available, otherwise use majority vote
    if 'gold_standard' in annotations.columns and annotations['gold_standard'].notna().any():
        gold_standard = pd.to_numeric(annotations['gold_standard'], errors='coerce')
        valid_mask = gold_standard.notna()
    else:
        print("⚠️ No gold standard found, using majority vote between annotators")
        annotations['annotator_1'] = pd.to_numeric(annotations['annotator_1'], errors='coerce')
        annotations['annotator_2'] = pd.to_numeric(annotations['annotator_2'], errors='coerce')
        
        # Majority vote (or use annotator_1 if disagreement)
        gold_standard = annotations.apply(
            lambda row: row['annotator_1'] if row['annotator_1'] == row['annotator_2'] 
            else row['annotator_1'], axis=1
        )
        valid_mask = gold_standard.notna()
    
    gold_standard = gold_standard[valid_mask].values
    
    # Model predictions: all matches above threshold are predicted as 1 (correct)
    # This assumes that the model predicted these as matches
    model_predictions = np.ones(len(gold_standard))
    
    if len(gold_standard) == 0:
        print("❌ No valid labels for evaluation")
        return {}
    
    # Calculate metrics
    precision = precision_score(gold_standard, model_predictions, zero_division=0)
    recall = recall_score(gold_standard, model_predictions, zero_division=0)
    f1 = f1_score(gold_standard, model_predictions, zero_division=0)
    
    # Calculate true/false positives/negatives
    tp = ((model_predictions == 1) & (gold_standard == 1)).sum()
    fp = ((model_predictions == 1) & (gold_standard == 0)).sum()
    tn = ((model_predictions == 0) & (gold_standard == 0)).sum()
    fn = ((model_predictions == 0) & (gold_standard == 1)).sum()
    
    results = {
        'method': method_name,
        'total_validated': len(gold_standard),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'accuracy': round((tp + tn) / len(gold_standard), 4) if len(gold_standard) > 0 else 0
    }
    
    # Print results
    print(f"\n📊 Validation Results:")
    print(f"   Total validated: {results['total_validated']}")
    print(f"   True Positives: {results['true_positives']}")
    print(f"   False Positives: {results['false_positives']}")
    print(f"   True Negatives: {results['true_negatives']}")
    print(f"   False Negatives: {results['false_negatives']}")
    print(f"\n   Precision: {results['precision']:.4f}")
    print(f"   Recall: {results['recall']:.4f}")
    print(f"   F1 Score: {results['f1_score']:.4f}")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    
    return results


def generate_validation_report(agreement_results: Dict, validation_results: Dict, 
                               output_dir: str = './reports') -> str:
    """
    Generate comprehensive validation report.
    """
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report_path = f"{output_dir}/validation_report_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(" MANUAL VALIDATION REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("1. INTER-ANNOTATOR AGREEMENT\n")
        f.write("-" * 100 + "\n")
        if agreement_results:
            f.write(f"Total annotations: {agreement_results['total_annotations']}\n")
            f.write(f"Agreements: {agreement_results['agreements']}\n")
            f.write(f"Disagreements: {agreement_results['disagreements']}\n")
            f.write(f"Agreement rate: {agreement_results['agreement_rate']:.2f}%\n")
            f.write(f"Cohen's Kappa: {agreement_results['cohens_kappa']:.4f} ({agreement_results['kappa_interpretation']})\n")
            
            if len(agreement_results.get('disagreement_cases', [])) > 0:
                f.write(f"\nDisagreement cases: {len(agreement_results['disagreement_cases'])}\n")
        f.write("\n")
        
        f.write("2. VALIDATION METRICS\n")
        f.write("-" * 100 + "\n")
        if validation_results:
            f.write(f"Method: {validation_results['method']}\n")
            f.write(f"Total validated: {validation_results['total_validated']}\n")
            f.write(f"True Positives: {validation_results['true_positives']}\n")
            f.write(f"False Positives: {validation_results['false_positives']}\n")
            f.write(f"True Negatives: {validation_results['true_negatives']}\n")
            f.write(f"False Negatives: {validation_results['false_negatives']}\n")
            f.write(f"\nPrecision: {validation_results['precision']:.4f}\n")
            f.write(f"Recall: {validation_results['recall']:.4f}\n")
            f.write(f"F1 Score: {validation_results['f1_score']:.4f}\n")
            f.write(f"Accuracy: {validation_results['accuracy']:.4f}\n")
        f.write("\n")
    
    print(f"\n💾 Validation report saved: {report_path}")
    return report_path


# ===================================================================
# MAIN FUNCTION: COMPREHENSIVE ANALYSIS (UPDATED)
# ===================================================================

def load_and_process_tables():
    """
    Main function with comprehensive analysis including:
    - Model configuration reporting
    - Advanced preprocessing (tokenization, stopwords, lemmatization)
    - Threshold sensitivity analysis
    - Ensemble weights ablation study
    """
   
    print("=" * 80)
    print(" 🧠 COMPREHENSIVE SEMANTIC SIMILARITY ANALYSIS")
    print("=" * 80)
   
    # ----------------------------------------------------------------
    # STEP 1: LOAD DATA
    # ----------------------------------------------------------------
    print("\n📂 STEP 1: Loading Excel files...")
   
    file_path = 'C:/Users/Diego/Downloads/article/Unit_SOC_ISCO.xlsx'
    
    try:
        table1 = pd.read_excel(file_path, sheet_name='ISCO-08')
        table2 = pd.read_excel(file_path, sheet_name='SOC-2020')
       
        print(f" ✅ Table 1 (ISCO-08): {len(table1)} occupations loaded")
        print(f" ✅ Table 2 (SOC-2020): {len(table2)} occupations loaded")
       
    except FileNotFoundError:
        print(f" ❌ ERROR: File not found at {file_path}")
        return None
    except ValueError as e:
        print(f" ❌ ERROR: {e}")
        return None
   
    # ----------------------------------------------------------------
    # STEP 2: CONFIGURE PARAMETERS
    # ----------------------------------------------------------------
    TEXT_COLUMN_TABLE1 = 'description'
    TEXT_COLUMN_TABLE2 = 'Group_Title'
    ID_COLUMN_TABLE1 = 'unit'
    ID_COLUMN_TABLE2 = 'Unit_Group'
   
    # ----------------------------------------------------------------
    # STEP 3: INITIALIZE MATCHER WITH ADVANCED PREPROCESSING
    # ----------------------------------------------------------------
    print("\n🔧 STEP 3: Initializing semantic matcher...")
    matcher = OccupationSemanticMatcher(
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        use_semantic_embeddings=True,
        cache_dir='./embeddings_cache',
        local_model_dir='./local_models',
        batch_size=32,
        seed=42,
        use_advanced_preprocessing=True,  # Enable tokenization, stopwords, lemmatization
        language='english'  # Change to 'spanish', 'french', etc. if needed
    )
    
    # Load model to extract metadata
    matcher.load_model()
    
    # Display model configuration
    print("\n" + "=" * 80)
    print(" 📋 MODEL CONFIGURATION REPORT")
    print("=" * 80)
    model_config = matcher.model_metadata
    print(f" 🤖 Model Variant: {model_config['model_variant']}")
    print(f" 🔄 Pooling Strategy: {model_config['pooling_strategy']}")
    print(f" 📏 Max Sequence Length: {model_config['max_sequence_length']}")
    print(f" 📊 Embedding Dimension: {model_config['embedding_dimension']}")
    print(f" 📦 Batch Size: {model_config['batch_size']}")
    print(f" 🎲 Random Seed: {model_config['random_seed']}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print(" 🔧 PREPROCESSING CONFIGURATION")
    print("=" * 80)
    print(f" 📝 Method: {'Advanced (Tokenization + Stopwords + Lemmatization)' if matcher.use_advanced_preprocessing else 'Basic'}")
    print(f" 🌐 Language: {matcher.language}")
    print(f" 🗑️ Stopwords loaded: {len(matcher.stop_words)}")
    print(f" 🔄 Lemmatizer: {'Enabled (WordNet)' if matcher.use_advanced_preprocessing else 'Disabled'}")
    print("=" * 80)
   
    # ----------------------------------------------------------------
    # STEP 4: THRESHOLD SENSITIVITY ANALYSIS
    # ----------------------------------------------------------------
    print("\n🔍 STEP 4: Threshold Sensitivity Analysis...")
    sensitivity_results = threshold_sensitivity_analysis(
        matcher, table1, table2,
        TEXT_COLUMN_TABLE1, TEXT_COLUMN_TABLE2,
        ID_COLUMN_TABLE1, ID_COLUMN_TABLE2,
        method='semantic'
    )
    
    # ----------------------------------------------------------------
    # STEP 4.5: INDIVIDUAL METHOD COMPARISON
    # ----------------------------------------------------------------
    print("\n📊 STEP 4.5: Individual Method Comparison (BoW, TF-IDF, Embeddings)...")
    print("=" * 80)
    print(" COMPARING INDIVIDUAL SIMILARITY METHODS")
    print("=" * 80)
    
    threshold_comparison = 0.3
    
    # Test BoW
    print(f"\n🔹 Testing Bag of Words (BoW) - Threshold: {threshold_comparison*100:.0f}%")
    bow_results = matcher.find_best_matches(
        df1=table1,
        df2=table2,
        text_column1=TEXT_COLUMN_TABLE1,
        text_column2=TEXT_COLUMN_TABLE2,
        id_column1=ID_COLUMN_TABLE1,
        id_column2=ID_COLUMN_TABLE2,
        threshold=threshold_comparison,
        top_n=3,
        method='bow'
    )
    
    if len(bow_results) > 0:
        bow_report = matcher.generate_similarity_report(bow_results)
        print(f"   ✅ Total matches: {bow_report['total_matches']}")
        print(f"   📈 Average similarity: {bow_report['average_similarity']:.2f}%")
        print(f"   📊 Median similarity: {bow_report['median_similarity']:.2f}%")
        print(f"   📉 Std deviation: {bow_report['std_similarity']:.2f}%")
        print(f"   🟢 High confidence (≥80%): {bow_report['high_confidence_matches']}")
        print(f"   🟡 Medium confidence (60-80%): {bow_report['medium_confidence_matches']}")
        print(f"   🔴 Low confidence (<60%): {bow_report['low_confidence_matches']}")
        
        # Show top 5 matches
        print(f"\n   📋 Top 5 Matches:")
        for idx, row in bow_results.head(5).iterrows():
            print(f"      {idx+1}. {row['occupation_table1']} ↔ {row['occupation_table2']} ({row['similarity_percentage']:.2f}%)")
    else:
        print(f"   ❌ No matches found")
    
    # Test TF-IDF
    print(f"\n🔹 Testing TF-IDF - Threshold: {threshold_comparison*100:.0f}%")
    tfidf_results = matcher.find_best_matches(
        df1=table1,
        df2=table2,
        text_column1=TEXT_COLUMN_TABLE1,
        text_column2=TEXT_COLUMN_TABLE2,
        id_column1=ID_COLUMN_TABLE1,
        id_column2=ID_COLUMN_TABLE2,
        threshold=threshold_comparison,
        top_n=3,
        method='tfidf'
    )
    
    if len(tfidf_results) > 0:
        tfidf_report = matcher.generate_similarity_report(tfidf_results)
        print(f"   ✅ Total matches: {tfidf_report['total_matches']}")
        print(f"   📈 Average similarity: {tfidf_report['average_similarity']:.2f}%")
        print(f"   📊 Median similarity: {tfidf_report['median_similarity']:.2f}%")
        print(f"   📉 Std deviation: {tfidf_report['std_similarity']:.2f}%")
        print(f"   🟢 High confidence (≥80%): {tfidf_report['high_confidence_matches']}")
        print(f"   🟡 Medium confidence (60-80%): {tfidf_report['medium_confidence_matches']}")
        print(f"   🔴 Low confidence (<60%): {tfidf_report['low_confidence_matches']}")
        
        # Show top 5 matches
        print(f"\n   📋 Top 5 Matches:")
        for idx, row in tfidf_results.head(5).iterrows():
            print(f"      {idx+1}. {row['occupation_table1']} ↔ {row['occupation_table2']} ({row['similarity_percentage']:.2f}%)")
    else:
        print(f"   ❌ No matches found")
    
    # Test Semantic Embeddings
    print(f"\n🔹 Testing Semantic Embeddings - Threshold: {threshold_comparison*100:.0f}%")
    semantic_results = matcher.find_best_matches(
        df1=table1,
        df2=table2,
        text_column1=TEXT_COLUMN_TABLE1,
        text_column2=TEXT_COLUMN_TABLE2,
        id_column1=ID_COLUMN_TABLE1,
        id_column2=ID_COLUMN_TABLE2,
        threshold=threshold_comparison,
        top_n=3,
        method='semantic'
    )
    
    if len(semantic_results) > 0:
        semantic_report = matcher.generate_similarity_report(semantic_results)
        print(f"   ✅ Total matches: {semantic_report['total_matches']}")
        print(f"   📈 Average similarity: {semantic_report['average_similarity']:.2f}%")
        print(f"   📊 Median similarity: {semantic_report['median_similarity']:.2f}%")
        print(f"   📉 Std deviation: {semantic_report['std_similarity']:.2f}%")
        print(f"   🟢 High confidence (≥80%): {semantic_report['high_confidence_matches']}")
        print(f"   🟡 Medium confidence (60-80%): {semantic_report['medium_confidence_matches']}")
        print(f"   🔴 Low confidence (<60%): {semantic_report['low_confidence_matches']}")
        
        # Show top 5 matches
        print(f"\n   📋 Top 5 Matches:")
        for idx, row in semantic_results.head(5).iterrows():
            print(f"      {idx+1}. {row['occupation_table1']} ↔ {row['occupation_table2']} ({row['similarity_percentage']:.2f}%)")
    else:
        print(f"   ❌ No matches found")
    
    # Comparison table
    print("\n" + "-" * 120)
    print(" 📊 METHOD COMPARISON SUMMARY")
    print("-" * 120)
    print(f"{'Method':<20} {'Matches':<10} {'Avg Sim':<10} {'Median':<10} {'Std Dev':<10} {'High':<8} {'Medium':<8} {'Low':<8}")
    print("-" * 120)
    
    if len(bow_results) > 0:
        print(f"{'Bag of Words':<20} {bow_report['total_matches']:<10} "
              f"{bow_report['average_similarity']:<10.2f} {bow_report['median_similarity']:<10.2f} "
              f"{bow_report['std_similarity']:<10.2f} {bow_report['high_confidence_matches']:<8} "
              f"{bow_report['medium_confidence_matches']:<8} {bow_report['low_confidence_matches']:<8}")
    else:
        print(f"{'Bag of Words':<20} {'0':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
    
    if len(tfidf_results) > 0:
        print(f"{'TF-IDF':<20} {tfidf_report['total_matches']:<10} "
              f"{tfidf_report['average_similarity']:<10.2f} {tfidf_report['median_similarity']:<10.2f} "
              f"{tfidf_report['std_similarity']:<10.2f} {tfidf_report['high_confidence_matches']:<8} "
              f"{tfidf_report['medium_confidence_matches']:<8} {tfidf_report['low_confidence_matches']:<8}")
    else:
        print(f"{'TF-IDF':<20} {'0':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
    
    if len(semantic_results) > 0:
        print(f"{'Semantic Embeddings':<20} {semantic_report['total_matches']:<10} "
              f"{semantic_report['average_similarity']:<10.2f} {semantic_report['median_similarity']:<10.2f} "
              f"{semantic_report['std_similarity']:<10.2f} {semantic_report['high_confidence_matches']:<8} "
              f"{semantic_report['medium_confidence_matches']:<8} {semantic_report['low_confidence_matches']:<8}")
    else:
        print(f"{'Semantic Embeddings':<20} {'0':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
    
    print("-" * 120)
    
    # Save individual method results
    print("\n💾 Saving individual method results...")
    output_base = 'C:/Users/Diego/Downloads/article/rev2/new/ISCOSOC/'
    
    if len(bow_results) > 0:
        bow_filename = f"{output_base}method_bow_results.csv"
        bow_results.to_csv(bow_filename, index=False, encoding='utf-8-sig')
        print(f"   ✅ BoW results saved: {bow_filename}")
    
    if len(tfidf_results) > 0:
        tfidf_filename = f"{output_base}method_tfidf_results.csv"
        tfidf_results.to_csv(tfidf_filename, index=False, encoding='utf-8-sig')
        print(f"   ✅ TF-IDF results saved: {tfidf_filename}")
    
    if len(semantic_results) > 0:
        semantic_filename = f"{output_base}method_semantic_results.csv"
        semantic_results.to_csv(semantic_filename, index=False, encoding='utf-8-sig')
        print(f"   ✅ Semantic results saved: {semantic_filename}")
    
    # Store method comparison results
    method_comparison_results = {
        'bow': {'results_df': bow_results, 'report': bow_report if len(bow_results) > 0 else None},
        'tfidf': {'results_df': tfidf_results, 'report': tfidf_report if len(tfidf_results) > 0 else None},
        'semantic': {'results_df': semantic_results, 'report': semantic_report if len(semantic_results) > 0 else None}
    }
    
    # ----------------------------------------------------------------
    # STEP 5: ENSEMBLE WEIGHTS ABLATION
    # ----------------------------------------------------------------
    print("\n🔬 STEP 5: Ensemble Weights Ablation Study...")
    ablation_results = ensemble_ablation_analysis(
        matcher, table1, table2,
        TEXT_COLUMN_TABLE1, TEXT_COLUMN_TABLE2,
        ID_COLUMN_TABLE1, ID_COLUMN_TABLE2,
        threshold=0.3
    )
    
    # ----------------------------------------------------------------
    # STEP 6: GENERATE REPORTS
    # ----------------------------------------------------------------
    print("\n💾 STEP 6: Generating comprehensive reports...")
    full_report = generate_comprehensive_report(
        matcher, sensitivity_results, ablation_results,
        output_dir='./reports'
    )
    
    # ----------------------------------------------------------------
    # STEP 7: DISPLAY RUNTIME SUMMARY
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" ⏱️ RUNTIME SUMMARY")
    print("=" * 80)
    runtime = matcher.runtime_stats
    print(f" Model Loading: {runtime.get('model_loading_time', 0):.2f}s")
    print(f" Text Preprocessing: {runtime.get('preprocessing_time', 0):.2f}s")
    print(f" Embedding Generation: {runtime.get('embedding_time', 0):.2f}s")
    print(f" Similarity Computation: {runtime.get('similarity_computation_time', 0):.2f}s")
    print(f" BoW Computation: {runtime.get('bow_computation_time', 0):.2f}s")
    print(f" TF-IDF Computation: {runtime.get('tfidf_computation_time', 0):.2f}s")
    print(f" Total Time: {runtime.get('total_time', 0):.2f}s")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print(" 📊 PREPROCESSING IMPACT SUMMARY")
    print("=" * 80)
    prep_stats = matcher.preprocessing_stats
    print(f" Texts processed: {prep_stats['total_texts_processed']}")
    print(f" Avg tokens before: {prep_stats['avg_tokens_per_text_before']}")
    print(f" Avg tokens after: {prep_stats['avg_tokens_per_text_after']}")
    print(f" Stopwords removed: {prep_stats['stopwords_removed']} ({prep_stats.get('stopwords_removal_rate', 0):.2f}%)")
    print(f" Token reduction rate: {prep_stats.get('token_reduction_rate', 0):.2f}%")
    print(f" Tokens lemmatized: {prep_stats['tokens_lemmatized']}")
    print("=" * 80)
    
    # ----------------------------------------------------------------
    # STEP 8: SAVE DETAILED RESULTS
    # ----------------------------------------------------------------
    print("\n💾 STEP 8: Saving detailed results...")
    output_base = 'C:/Users/Diego/Downloads/article/rev2/new/ISCOSOC/'
    
    # Save threshold sensitivity results
    for thresh_name, data in sensitivity_results.items():
        if 'results_df' in data and len(data['results_df']) > 0:
            filename = f"{output_base}threshold_{thresh_name}_results.csv"
            data['results_df'].to_csv(filename, index=False, encoding='utf-8-sig')
            print(f" ✅ Threshold {thresh_name}: {filename}")
    
    # Save ablation results
    for config_name, data in ablation_results.items():
        if 'results_df' in data and len(data['results_df']) > 0:
            safe_name = config_name.replace('/', '_').replace(' ', '_')
            filename = f"{output_base}ablation_{safe_name}.csv"
            data['results_df'].to_csv(filename, index=False, encoding='utf-8-sig')
            print(f" ✅ {config_name}: {filename}")
    
    print("\n" + "=" * 80)
    print(" ✅ COMPREHENSIVE ANALYSIS COMPLETED")
    print("=" * 80)
    print("\n📊 Key Findings:")
    print(" 1. Model configuration and metadata extracted")
    print(" 2. Advanced preprocessing applied (tokenization, stopwords, lemmatization)")
    print(" 3. Threshold sensitivity analyzed (30%, 40%, 50%)")
    print(" 4. Individual methods compared (BoW, TF-IDF, Semantic)")
    print(" 5. Ensemble weights ablation completed (8 configurations)")
    print(" 6. Runtime statistics recorded")
    print(" 7. Comprehensive reports generated")
    
    # ----------------------------------------------------------------
    # STEP 9: CREATE VALIDATION SAMPLE (OPTIONAL)
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 📝 STEP 9: MANUAL VALIDATION SETUP")
    print("=" * 80)
    
    # Choose which results to validate (e.g., semantic results)
    if len(semantic_results) > 0:
        print("\n🎯 Creating validation sample from semantic embeddings results...")
        validation_sample = create_validation_sample(
            semantic_results, 
            sample_size=100, 
            stratified=True,
            random_state=42
        )
        
        # Save validation sample
        validation_sample_path = f"{output_base}validation_sample_for_annotation.csv"
        validation_sample.to_csv(validation_sample_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 Validation sample saved: {validation_sample_path}")
        print("\n📋 INSTRUCTIONS FOR MANUAL ANNOTATION:")
        print("   1. Open the validation sample file")
        print("   2. For each row, annotators should fill:")
        print("      - 'annotator_1': 1 if match is correct, 0 if incorrect")
        print("      - 'annotator_2': 1 if match is correct, 0 if incorrect")
        print("      - 'gold_standard': Final agreed label (after discussion)")
        print("      - 'notes': Any comments about the match")
        print("   3. Save the annotated file")
        print("   4. Run the validation analysis (see example below)")
        
        # Example code for validation analysis
        print("\n💡 To analyze annotations after completion, run:")
        print("   ```python")
        print(f"   annotations = load_annotations('{validation_sample_path}')")
        print("   agreement = calculate_inter_annotator_agreement(annotations)")
        print("   metrics = calculate_validation_metrics(annotations, 'Semantic Embeddings')")
        print("   generate_validation_report(agreement, metrics)")
        print("   ```")
    else:
        print("\n⚠️ No semantic results available for validation sample")
    
    return {
        'matcher': matcher,
        'sensitivity_results': sensitivity_results,
        'method_comparison': method_comparison_results,
        'ablation_results': ablation_results,
        'full_report': full_report,
        'validation_sample': validation_sample if len(semantic_results) > 0 else None,
        'table1': table1,
        'table2': table2
    }


# ===================================================================
# RUN THE COMPREHENSIVE ANALYSIS
# ===================================================================

if __name__ == "__main__":
    print("\n🚀 Starting comprehensive semantic analysis...")
    print("⏱️ Note: First run will take longer (model download)")
    print("💾 Subsequent runs will be faster (cache)\n")
   
    results = load_and_process_tables()
   
    if results is not None:
        print("\n" + "=" * 80)
        print(" 📚 RESULTS ACCESS GUIDE")
        print("=" * 80)
        print(" - results['matcher']: Main matcher object with model info")
        print(" - results['sensitivity_results']: Threshold analysis (30/40/50%)")
        print(" - results['method_comparison']: Individual method comparison (BoW/TF-IDF/Semantic)")
        print(" - results['ablation_results']: Ensemble weights experiments")
        print(" - results['full_report']: Complete JSON report")
        print(" - Check './reports/' folder for detailed reports")
        print(" - Check output folder for method_*.csv files")
        print("=" * 80)

# Cargar el archivo YA ANOTADO
annotations = pd.read_csv('C:/Users/Diego/Downloads/validation_sample_for_annotation_iscosoc.csv', encoding='utf-8-sig',sep=';')

# Calcular acuerdo entre anotadores
agreement = calculate_inter_annotator_agreement(annotations)

# Calcular métricas de validación
if agreement:  # Solo si hay anotaciones válidas
    metrics = calculate_validation_metrics(annotations, 'Semantic Embeddings')
    
    # Generar reporte
    if metrics:
        generate_validation_report(agreement, metrics)