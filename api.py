from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import numpy as np
import os
import io
import base64
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
import tempfile
import uuid
from datetime import datetime
from PIL import Image
from vedo import Mesh, show, colors as vcolors
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename
import logging
from functools import lru_cache, wraps
from typing import Dict, Tuple, Optional, Any, List
import concurrent.futures
import threading
from collections import defaultdict
from dataclasses import dataclass, asdict
import gc
import psutil
import time
from contextlib import contextmanager

# Configuration optimisée du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('star_api.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Configuration Flask optimisée
app = Flask(__name__)
CORS(app, origins=['*'], methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

# Configuration des constantes
@dataclass
class Config:
    BASE_DIR: str = os.path.join(os.path.dirname(__file__), '..', 'star_1_1')
    MAPPING_PATH: str = os.path.join(BASE_DIR, 'star_measurements_mapping.json')
    TEMP_DIR: str = tempfile.mkdtemp()
    UPLOAD_FOLDER: str = 'uploads'
    GENERATED_FOLDER: str = 'generated_clothes'
    ALLOWED_EXTENSIONS: set = None
    MAX_CONTENT_LENGTH: int = 32 * 1024 * 1024  # 32MB
    CACHE_SIZE: int = 128
    THREAD_POOL_SIZE: int = min(8, os.cpu_count() or 4)
    MEMORY_LIMIT_MB: int = 1024
    
    def __post_init__(self):
        self.ALLOWED_EXTENSIONS = {'obj', 'npz', 'ply'}
        # Création des dossiers nécessaires
        for folder in [self.UPLOAD_FOLDER, self.GENERATED_FOLDER]:
            os.makedirs(folder, exist_ok=True)

config = Config()

app.config.update({
    'UPLOAD_FOLDER': config.UPLOAD_FOLDER,
    'GENERATED_FOLDER': config.GENERATED_FOLDER,
    'MAX_CONTENT_LENGTH': config.MAX_CONTENT_LENGTH,
    'JSON_SORT_KEYS': False  # Optimisation JSON
})

# Pool de threads global pour les opérations parallèles
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=config.THREAD_POOL_SIZE)

# Cache thread-safe avec limitation de mémoire
class ThreadSafeCache:
    def __init__(self, max_size=config.CACHE_SIZE, max_memory_mb=config.MEMORY_LIMIT_MB):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.lock = threading.RLock()
        self._memory_check_counter = 0
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            # Vérification périodique de la mémoire
            self._memory_check_counter += 1
            if self._memory_check_counter % 10 == 0:
                self._cleanup_memory()
            
            # Nettoyage LRU si nécessaire
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def _evict_lru(self):
        if not self.access_times:
            return
        lru_key = min(self.access_times.keys(), key=self.access_times.get)
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def _cleanup_memory(self):
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > self.max_memory_mb:
            # Nettoyer 25% du cache
            cleanup_count = max(1, len(self.cache) // 4)
            sorted_keys = sorted(self.access_times.keys(), key=self.access_times.get)
            for key in sorted_keys[:cleanup_count]:
                del self.cache[key]
                del self.access_times[key]
            gc.collect()  # Force garbage collection
            logger.warning(f"Cache nettoyé: mémoire {memory_mb:.1f}MB > {self.max_memory_mb}MB")
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            gc.collect()

# Caches optimisés
models_cache = ThreadSafeCache(max_size=32, max_memory_mb=512)
measurements_cache = ThreadSafeCache(max_size=64, max_memory_mb=128)
clothing_cache = ThreadSafeCache(max_size=128, max_memory_mb=256)

# Optimisation des couleurs et vêtements avec numpy arrays
COULEURS_RGB = np.array([
    [25, 25, 25],           # Noir
    [25, 25, 128],          # Bleu Marine
    [77, 77, 77],           # Gris Anthracite
    [128, 25, 51],          # Bordeaux
    [204, 25, 25],          # Rouge
    [77, 102, 204],         # Bleu Clair
    [102, 128, 51],         # Vert Olive
    [102, 77, 51],          # Marron
    [230, 230, 217],        # Blanc Cassé
    [204, 153, 179],        # Rose Poudré
], dtype=np.uint8)

COULEURS_NOMS = [
    "Noir", "Bleu Marine", "Gris Anthracite", "Bordeaux", "Rouge",
    "Bleu Clair", "Vert Olive", "Marron", "Blanc Cassé", "Rose Poudré"
]

COULEURS_DISPONIBLES = dict(zip(COULEURS_NOMS, COULEURS_RGB.tolist()))

# Structure optimisée pour les types de vêtements
@dataclass
class ClothingParams:
    categorie: str
    type: str
    longueur_relative: float
    description: str
    ampleur: Optional[float] = None
    evasement: Optional[float] = None

TYPES_VETEMENTS = {
    # Jupes droites - optimisées
    "Mini-jupe droite": ClothingParams("jupe", "droite", 0.15, "Mini-jupe droite (mi-cuisse)"),
    "Jupe droite au genou": ClothingParams("jupe", "droite", 0.35, "Jupe droite classique (genou)"),
    "Jupe droite longue": ClothingParams("jupe", "droite", 0.75, "Jupe droite longue (cheville)"),
    
    # Jupes ovales
    "Mini-jupe ovale": ClothingParams("jupe", "ovale", 0.15, "Mini-jupe ovale évasée (mi-cuisse)", ampleur=1.3),
    "Jupe ovale au genou": ClothingParams("jupe", "ovale", 0.35, "Jupe ovale classique (genou)", ampleur=1.4),
    "Jupe ovale longue": ClothingParams("jupe", "ovale", 0.75, "Jupe ovale longue (cheville)", ampleur=1.5),
    "Jupe ovale bouffante": ClothingParams("jupe", "ovale", 0.35, "Jupe ovale très évasée (style bouffant)", ampleur=1.8),
    
    # Jupes trapèze
    "Mini-jupe trapèze": ClothingParams("jupe", "trapeze", 0.15, "Mini-jupe trapèze évasée (mi-cuisse)", evasement=1.4),
    "Jupe trapèze au genou": ClothingParams("jupe", "trapeze", 0.35, "Jupe trapèze classique (genou)", evasement=1.6),
    "Jupe trapèze longue": ClothingParams("jupe", "trapeze", 0.75, "Jupe trapèze longue (cheville)", evasement=1.8),
    "Jupe trapèze évasée": ClothingParams("jupe", "trapeze", 0.45, "Jupe trapèze très évasée (style A-line)", evasement=2.0),
}

# Décorateurs utilitaires optimisés
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.debug(f"{func.__name__} executed in {duration:.3f}s")
        return result
    return wrapper

def memory_monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_diff = mem_after - mem_before
        if mem_diff > 50:  # Plus de 50MB de différence
            logger.warning(f"{func.__name__} used {mem_diff:.1f}MB memory")
        return result
    return wrapper

@contextmanager
def error_handler(operation_name="Operation"):
    try:
        start_time = time.perf_counter()
        yield
        duration = time.perf_counter() - start_time
        logger.debug(f"{operation_name} completed in {duration:.3f}s")
    except Exception as e:
        logger.error(f"{operation_name} failed: {str(e)}")
        raise

# Classe principale ultra-optimisée
class STARClothingGeneratorOptimized:
    """Générateur STAR optimisé pour haute performance"""
    
    def __init__(self):
        self.mapping_cache_data = None
        self._load_mapping_lazy()
        self.computation_cache = {}
        self.vectorized_functions = self._setup_vectorized_functions()
        logger.info("STARClothingGenerator optimisé initialisé")
    
    def _load_mapping_lazy(self):
        """Chargement paresseux du mapping"""
        def _load():
            if self.mapping_cache_data is None:
                with open(config.MAPPING_PATH, 'r', encoding='utf-8') as f:
                    self.mapping_cache_data = json.load(f)
            return self.mapping_cache_data
        return _load
    
    def _setup_vectorized_functions(self):
        """Configuration des fonctions vectorisées"""
        return {
            'distance_calculation': np.vectorize(self._calculate_distance_optimized),
            'radius_calculation': self._calculate_radius_vectorized
        }
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def _calculate_distance_optimized(p1: tuple, p2: tuple) -> float:
        """Calcul de distance optimisé avec cache"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    @staticmethod
    def _calculate_radius_vectorized(vertices: np.ndarray, y_target: float, tolerance: float = 0.05) -> float:
        """Calcul de rayon vectorisé ultra-rapide"""
        y_vals = vertices[:, 1]
        mask = np.abs(y_vals - y_target) < tolerance
        
        if not np.any(mask):
            return 0.1
        
        points = vertices[mask]
        distances = np.sqrt(np.sum(points[:, [0, 2]] ** 2, axis=1))
        return np.percentile(distances, 75) if len(distances) > 0 else 0.1
    
    @timing_decorator
    @memory_monitor
    def charger_modele_star_optimized(self, npz_path: str) -> Dict[str, Any]:
        """Chargement de modèle STAR ultra-optimisé"""
        cache_key = f"model_{npz_path}_{os.path.getmtime(npz_path)}"
        
        cached_model = models_cache.get(cache_key)
        if cached_model:
            logger.info(f"Modèle chargé depuis le cache: {npz_path}")
            return cached_model
        
        with error_handler(f"Chargement modèle {npz_path}"):
            # Chargement optimisé avec mmap pour les gros fichiers
            data = np.load(npz_path, mmap_mode='r' if os.path.getsize(npz_path) > 100*1024*1024 else None)
            
            model_data = {
                'v_template': np.array(data['v_template'], dtype=np.float32),  # Float32 pour économiser la mémoire
                'f': np.array(data['f'], dtype=np.uint32),
                'J_regressor': np.array(data['J_regressor'], dtype=np.float32) if 'J_regressor' in data else None,
                'shapedirs': np.array(data['shapedirs'], dtype=np.float32) if 'shapedirs' in data else None,
                'posedirs': np.array(data['posedirs'], dtype=np.float32) if 'posedirs' in data else None,
                'metadata': {
                    'file_path': npz_path,
                    'loaded_at': time.time(),
                    'file_size': os.path.getsize(npz_path)
                }
            }
            
            # Calcul des joints si J_regressor existe
            if model_data['J_regressor'] is not None:
                model_data['Jtr'] = model_data['J_regressor'].dot(model_data['v_template'])
            
            models_cache.set(cache_key, model_data)
            logger.info(f"Modèle chargé et mis en cache: {npz_path} ({os.path.getsize(npz_path)/1024/1024:.1f}MB)")
            return model_data
    
    @timing_decorator
    def calculer_mesures_modele_ultra_optimized(self, vertices: np.ndarray, joints: np.ndarray, mapping: Dict[str, Any]) -> Dict[str, float]:
        """Calcul de mesures ultra-optimisé avec parallélisation intelligente"""
        cache_key = f"measures_{hash(vertices.tobytes())}_{hash(str(mapping))}"
        
        cached_measures = measurements_cache.get(cache_key)
        if cached_measures:
            return cached_measures
        
        # Pré-calculs vectorisés
        vertex_distances_precomputed = {}
        
        def calculate_single_measure(item):
            mesure, info = item
            joint_indices = info["joints"]
            
            if len(joint_indices) == 2:
                # Distance directe entre deux joints
                return mesure, float(np.linalg.norm(joints[joint_indices[1]] - joints[joint_indices[0]]))
            
            elif len(joint_indices) == 1:
                joint_pos = joints[joint_indices[0]]
                
                # Utiliser le cache de distances pré-calculées
                cache_key_local = joint_indices[0]
                if cache_key_local not in vertex_distances_precomputed:
                    distances = np.linalg.norm(vertices - joint_pos, axis=1)
                    vertex_distances_precomputed[cache_key_local] = distances
                else:
                    distances = vertex_distances_precomputed[cache_key_local]
                
                # Optimisation: utiliser percentile au lieu de moyennes
                threshold = np.percentile(distances, 20)
                nearby_indices = distances < threshold
                
                if np.sum(nearby_indices) > 3:
                    nearby_vertices = vertices[nearby_indices]
                    center = np.mean(nearby_vertices, axis=0)
                    radii = np.linalg.norm(nearby_vertices - center, axis=1)
                    return mesure, float(2 * np.pi * np.mean(radii))
                else:
                    return mesure, 50.0
            
            return mesure, 0.0
        
        # Parallélisation adaptative
        mapping_items = list(mapping.items())
        if len(mapping_items) > 20:
            # Utiliser le thread pool pour les gros modèles
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(mapping_items))) as executor:
                results = list(executor.map(calculate_single_measure, mapping_items))
        else:
            # Séquentiel pour les petits modèles (éviter l'overhead)
            results = [calculate_single_measure(item) for item in mapping_items]
        
        mesures = dict(results)
        measurements_cache.set(cache_key, mesures)
        return mesures
    
    @timing_decorator
    def detecter_points_anatomiques_ultra_optimized(self, verts: np.ndarray) -> Dict[str, float]:
        """Détection de points anatomiques ultra-rapide"""
        # Calculs vectorisés en une seule passe
        y_vals = verts[:, 1]
        y_stats = {
            'max': np.max(y_vals),
            'min': np.min(y_vals),
        }
        y_stats['height'] = y_stats['max'] - y_stats['min']
        
        # Points anatomiques calculés vectoriellement
        anatomical_ratios = np.array([0.1, 0.2, 0.3, 0.45, 0.75])
        y_points = y_stats['max'] - anatomical_ratios * y_stats['height']
        
        # Calcul des rayons en parallèle avec numpy
        radii = np.array([self._calculate_radius_vectorized(verts, y) for y in y_points[:4]])
        
        # Détection optimisée des bras avec masquage vectoriel
        arm_zone_mask = (y_vals <= y_points[1]) & (y_vals >= y_points[3])
        if np.any(arm_zone_mask):
            arm_points = verts[arm_zone_mask]
            arm_distances = np.sqrt(np.sum(arm_points[:, [0, 2]] ** 2, axis=1))
            arm_threshold = np.percentile(arm_distances, 85) if len(arm_distances) > 0 else 0.3
        else:
            arm_threshold = 0.3
        
        return {
            'y_tete': float(y_points[0]),
            'y_epaules': float(y_points[1]),
            'y_taille': float(y_points[2]),
            'y_hanches': float(y_points[3]),
            'y_genoux': float(y_points[4]),
            'y_min': float(y_stats['min']),
            'y_max': float(y_stats['max']),
            'hauteur_totale': float(y_stats['height']),
            'rayon_tete': float(radii[0]),
            'rayon_epaules': float(radii[1]),
            'rayon_taille': float(radii[2]),
            'rayon_hanches': float(radii[3]),
            'seuil_bras': float(arm_threshold)
        }
    
    @timing_decorator
    def deformer_modele_ultra_optimized(self, v_template: np.ndarray, shapedirs: Optional[np.ndarray],
                                      mesures_cibles: Dict[str, float], mesures_actuelles: Dict[str, float],
                                      J_regressor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Déformation de modèle ultra-optimisée"""
        if shapedirs is None:
            return v_template, np.zeros(10, dtype=np.float32)
        
        n_betas = min(10, shapedirs.shape[2])
        
        # Pré-calculs optimisés
        mesures_communes = list(set(mesures_cibles.keys()) & set(mesures_actuelles.keys()))
        if not mesures_communes:
            return v_template, np.zeros(n_betas, dtype=np.float32)
        
        # Calculs vectorisés des ratios
        ratios_array = np.array([
            mesures_cibles[m] / max(mesures_actuelles[m], 1e-6) for m in mesures_communes
        ], dtype=np.float32)
        targets_array = np.array([mesures_cibles[m] for m in mesures_communes], dtype=np.float32)
        
        # Fonction objectif optimisée avec numba-like operations
        def objective_ultra_optimized(betas):
            # Déformation vectorisée
            deformation = np.tensordot(shapedirs[:, :, :n_betas], betas, axes=([2], [0]))
            vertices_deformed = v_template + deformation
            
            # Calcul d'erreur vectorisé
            current_values = np.array([mesures_actuelles[m] for m in mesures_communes], dtype=np.float32)
            scaled_values = current_values * ratios_array
            error = np.sum((scaled_values - targets_array) ** 2)
            
            # Régularisation L2
            regularization = 0.1 * np.sum(betas ** 2)
            return float(error + regularization)
        
        # Optimisation avec bornes serrées
        initial_betas = np.zeros(n_betas, dtype=np.float32)
        bounds = [(-2.5, 2.5)] * n_betas  # Bornes plus serrées pour stabilité
        
        # Optimisation avec méthode adaptée
        result = minimize(
            objective_ultra_optimized, 
            initial_betas, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        optimal_betas = result.x.astype(np.float32)
        
        # Application finale de la déformation
        final_deformation = np.tensordot(shapedirs[:, :, :n_betas], optimal_betas, axes=([2], [0]))
        vertices_final = v_template + final_deformation
        
        return vertices_final, optimal_betas
    
    @timing_decorator
    def calculer_profil_jupe_optimized(self, points_anat: Dict[str, float], type_jupe: str,
                                     longueur_relative: float, **kwargs) -> Dict[str, Any]:
        """Calcul de profil de jupe optimisé avec fonctions pré-compilées"""
        # Extraction optimisée des paramètres
        y_taille, y_hanches, y_min = points_anat['y_taille'], points_anat['y_hanches'], points_anat['y_min']
        hauteur_totale = points_anat['hauteur_totale']
        rayon_taille, rayon_hanches = points_anat['rayon_taille'], points_anat['rayon_hanches']
        
        # Calculs de base vectorisés
        y_debut_jupe = y_taille - 0.03
        y_bas_jupe = max(y_hanches - (longueur_relative * hauteur_totale), y_min + 0.1)
        rayon_debut = rayon_taille * 0.88
        
        # Fonctions spécialisées pour chaque type
        if type_jupe == "droite":
            rayon_hanches_jupe = rayon_hanches * 0.95
            rayon_bas = rayon_hanches_jupe * 1.02
            
            def rayon_fonction(y):
                y = np.atleast_1d(y)
                result = np.zeros_like(y)
                
                # Conditions vectorisées
                mask1 = y <= y_debut_jupe
                mask2 = mask1 & (y >= y_hanches)
                mask3 = mask1 & (y < y_hanches) & (y >= y_bas_jupe)
                
                # Calculs vectorisés
                t2 = np.clip((y_debut_jupe - y[mask2]) / max(y_debut_jupe - y_hanches, 0.001), 0, 1)
                result[mask2] = rayon_debut + t2 * (rayon_hanches_jupe - rayon_debut)
                
                t3 = np.clip((y_hanches - y[mask3]) / max(y_hanches - y_bas_jupe, 0.001), 0, 1)
                result[mask3] = rayon_hanches_jupe + t3 * (rayon_bas - rayon_hanches_jupe)
                
                return result if result.shape else float(result)
                
        elif type_jupe == "ovale":
            ampleur = kwargs.get('ampleur', 1.4)
            rayon_max = rayon_hanches * ampleur
            rayon_bas = rayon_hanches * 0.9
            y_max_largeur = y_hanches - 0.1
            
            def rayon_fonction(y):
                y = np.atleast_1d(y)
                result = np.zeros_like(y)
                
                mask1 = y <= y_debut_jupe
                mask2 = mask1 & (y >= y_max_largeur)
                mask3 = mask1 & (y < y_max_largeur) & (y >= y_bas_jupe)
                
                t2 = np.clip((y_debut_jupe - y[mask2]) / max(y_debut_jupe - y_max_largeur, 0.001), 0, 1)
                t_curve2 = 0.5 * (1 - np.cos(np.pi * t2))
                result[mask2] = rayon_debut + t_curve2 * (rayon_max - rayon_debut)
                
                t3 = np.clip((y_max_largeur - y[mask3]) / max(y_max_largeur - y_bas_jupe, 0.001), 0, 1)
                t_curve3 = 0.5 * (1 + np.cos(np.pi * t3))
                result[mask3] = rayon_max - (rayon_max - rayon_bas) * (1 - t_curve3)
                
                return result if result.shape else float(result)
                
        elif type_jupe == "trapeze":
            evasement = kwargs.get('evasement', 1.6)
            rayon_bas = rayon_hanches * evasement
            
            def rayon_fonction(y):
                y = np.atleast_1d(y)
                result = np.zeros_like(y)
                
                mask = (y <= y_debut_jupe) & (y >= y_bas_jupe)
                t = np.clip((y_debut_jupe - y[mask]) / max(y_debut_jupe - y_bas_jupe, 0.001), 0, 1)
                result[mask] = rayon_debut + t * (rayon_bas - rayon_debut)
                
                return result if result.shape else float(result)
        
        else:
            def rayon_fonction(y):
                return 0.0
        
        return {
            'y_debut': y_debut_jupe,
            'y_bas': y_bas_jupe,
            'rayon_fonction': rayon_fonction,
            'type': type_jupe,
            'computed_at': time.time()
        }
    
    @timing_decorator
    def appliquer_vetement_ultra_optimized(self, verts: np.ndarray, profil_vetement: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Application de vêtement ultra-optimisée avec vectorisation complète"""
        verts_modifies = verts.copy()
        y_vals = verts[:, 1]
        
        # Masque vectorisé ultra-rapide
        masque_vetement = (y_vals <= profil_vetement['y_debut']) & (y_vals >= profil_vetement['y_bas'])
        indices_vetement = np.where(masque_vetement)[0]
        
        if len(indices_vetement) == 0:
            return verts_modifies, masque_vetement
        
        # Calculs entièrement vectorisés
        points_vetement = verts[indices_vetement]
        distances_actuelles = np.sqrt(np.sum(points_vetement[:, [0, 2]] ** 2, axis=1))
        
        # Application vectorisée de la fonction rayon
        y_points = points_vetement[:, 1]
        nouveaux_rayons = profil_vetement['rayon_fonction'](y_points)
        
        # Masque de validation vectorisé
        mask_valide = (distances_actuelles > 1e-6) & (nouveaux_rayons > 1e-6)
        indices_valides = indices_vetement[mask_valide]
        
        if len(indices_valides) > 0:
            # Facteurs de mise à l'échelle vectorisés
            facteurs = nouveaux_rayons[mask_valide] / distances_actuelles[mask_valide]
            
            # Application vectorisée
            verts_modifies[indices_valides, 0] *= facteurs
            verts_modifies[indices_valides, 2] *= facteurs
        
        return verts_modifies, masque_vetement
    
    @timing_decorator
    def generer_vetement_complet_ultra_optimized(self, npz_path: str, mesures_cibles: Dict[str, float],
                                               type_vetement: str, couleur: str) -> Dict[str, Any]:
        """Génération complète de vêtement ultra-optimisée"""
        try:
            # Validation des entrées
            if type_vetement not in TYPES_VETEMENTS:
                raise ValueError(f"Type de vêtement non supporté: {type_vetement}")
            if couleur not in COULEURS_DISPONIBLES:
                raise ValueError(f"Couleur non disponible: {couleur}")
            
            clothing_params = TYPES_VETEMENTS[type_vetement]
            couleur_rgb = COULEURS_DISPONIBLES[couleur]
            
            # Chargement du modèle avec cache
            model_data = self.charger_modele_star_optimized(npz_path)
            v_template = model_data['v_template']
            f = model_data['f']
            shapedirs = model_data.get('shapedirs')
            J_regressor = model_data.get('J_regressor')
            
            # Calcul des joints
            if J_regressor is not None:
                joints = J_regressor.dot(v_template)
            else:
                joints = np.zeros((24, 3), dtype=np.float32)
            
            # Calcul des mesures actuelles avec cache
            if self.mapping_cache_data is None:
                self._load_mapping_lazy()()
            
            mesures_actuelles = self.calculer_mesures_modele_ultra_optimized(
                v_template, joints, self.mapping_cache_data
            )
            
            # Déformation du modèle
            vertices_deformed, betas_optimaux = self.deformer_modele_ultra_optimized(
                v_template, shapedirs, mesures_cibles, mesures_actuelles, J_regressor
            )
            
            # Recalcul des joints après déformation
            if J_regressor is not None:
                joints_deformed = J_regressor.dot(vertices_deformed)
            else:
                joints_deformed = joints
            
            # Détection des points anatomiques
            points_anatomiques = self.detecter_points_anatomiques_ultra_optimized(vertices_deformed)
            
            # Calcul du profil de vêtement
            profil_params = {
                'ampleur': clothing_params.ampleur,
                'evasement': clothing_params.evasement
            }
            profil_params = {k: v for k, v in profil_params.items() if v is not None}
            
            profil_vetement = self.calculer_profil_jupe_optimized(
                points_anatomiques, 
                clothing_params.type,
                clothing_params.longueur_relative,
                **profil_params
            )
            
            # Application du vêtement
            vertices_with_clothing, masque_vetement = self.appliquer_vetement_ultra_optimized(
                vertices_deformed, profil_vetement
            )
            
            # Calcul des mesures finales
            mesures_finales = self.calculer_mesures_modele_ultra_optimized(
                vertices_with_clothing, joints_deformed, self.mapping_cache_data
            )
            
            # Génération de l'ID unique
            generation_id = str(uuid.uuid4())
            
            return {
                'id': generation_id,
                'vertices': vertices_with_clothing.astype(np.float32),
                'faces': f,
                'couleur_rgb': couleur_rgb,
                'mesures_finales': mesures_finales,
                'mesures_cibles': mesures_cibles,
                'type_vetement': type_vetement,
                'couleur': couleur,
                'betas_optimaux': betas_optimaux.tolist(),
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'clothing_params': asdict(clothing_params),
                    'profil_vetement_summary': {
                        'y_debut': profil_vetement['y_debut'],
                        'y_bas': profil_vetement['y_bas'],
                        'type': profil_vetement['type']
                    },
                    'nb_vertices_affected': int(np.sum(masque_vetement)),
                    'processing_stats': {
                        'model_file': os.path.basename(npz_path),
                        'model_vertices': len(vertices_with_clothing),
                        'model_faces': len(f)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de vêtement: {str(e)}")
            raise
    
    @timing_decorator
    def generer_mesh_vedo_optimized(self, vertices: np.ndarray, faces: np.ndarray, 
                                  couleur_rgb: List[int]) -> Mesh:
        """Génération de mesh Vedo optimisée"""
        try:
            # Normalisation des couleurs
            couleur_normalized = [c / 255.0 for c in couleur_rgb]
            
            # Création du mesh optimisée
            mesh = Mesh([vertices, faces])
            mesh.color(couleur_normalized)
            
            # Optimisations visuelles
            mesh.flat()  # Rendu plat pour les vêtements
            mesh.lighting('off')  # Désactiver l'éclairage pour de meilleures performances
            
            return mesh
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du mesh: {str(e)}")
            raise
    
    @timing_decorator
    def sauvegarder_fichier_optimized(self, vertices: np.ndarray, faces: np.ndarray, 
                                    filepath: str, format_fichier: str = 'obj') -> bool:
        """Sauvegarde de fichier optimisée"""
        try:
            if format_fichier.lower() == 'obj':
                with open(filepath, 'w', encoding='utf-8') as f:
                    # Écriture optimisée avec buffer
                    vertex_lines = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in vertices]
                    f.writelines(vertex_lines)
                    
                    face_lines = [f"f {face[0]+1} {face[1]+1} {face[2]+1}\n" for face in faces]
                    f.writelines(face_lines)
            
            elif format_fichier.lower() == 'ply':
                # Sauvegarde PLY optimisée
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {len(vertices)}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write(f"element face {len(faces)}\n")
                    f.write("property list uchar int vertex_indices\n")
                    f.write("end_header\n")
                    
                    # Écriture des vertices
                    for vertex in vertices:
                        f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                    
                    # Écriture des faces
                    for face in faces:
                        f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {str(e)}")
            return False

# Instance globale optimisée
generator = STARClothingGeneratorOptimized()

# Utilitaires Flask optimisés
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def secure_filename_custom(filename: str) -> str:
    """Version optimisée de secure_filename"""
    filename = secure_filename(filename)
    timestamp = int(time.time())
    return f"{timestamp}_{filename}"

def validate_measurements(measurements: Dict[str, Any]) -> Dict[str, float]:
    """Validation optimisée des mesures"""
    validated = {}
    for key, value in measurements.items():
        try:
            float_value = float(value)
            if 0 < float_value < 300:  # Limites raisonnables en cm
                validated[key] = float_value
            else:
                logger.warning(f"Mesure {key} ignorée: valeur {float_value} hors limites")
        except (ValueError, TypeError):
            logger.warning(f"Mesure {key} ignorée: conversion impossible")
    return validated

def create_error_response(message: str, status_code: int = 400) -> tuple:
    """Création de réponse d'erreur standardisée"""
    return jsonify({
        'success': False,
        'error': message,
        'timestamp': datetime.now().isoformat()
    }), status_code

def create_success_response(data: Any, message: str = "Opération réussie") -> dict:
    """Création de réponse de succès standardisée"""
    return jsonify({
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    })

# Routes de l'API optimisées
@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérification de l'état de l'API"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return create_success_response({
        'status': 'healthy',
        'version': '1.1.0-optimized',
        'memory_usage_mb': round(memory_info.rss / 1024 / 1024, 2),
        'cache_stats': {
            'models_cache_size': len(models_cache.cache),
            'measurements_cache_size': len(measurements_cache.cache),
            'clothing_cache_size': len(clothing_cache.cache)
        },
        'available_clothing_types': list(TYPES_VETEMENTS.keys()),
        'available_colors': list(COULEURS_DISPONIBLES.keys())
    })

@app.route('/api/types-vetements', methods=['GET'])
def get_types_vetements():
    """Récupération des types de vêtements disponibles"""
    types_details = {}
    for nom, params in TYPES_VETEMENTS.items():
        types_details[nom] = {
            'categorie': params.categorie,
            'type': params.type,
            'description': params.description,
            'longueur_relative': params.longueur_relative
        }
    
    return create_success_response(types_details)

@app.route('/api/couleurs', methods=['GET'])
def get_couleurs():
    """Récupération des couleurs disponibles"""
    return create_success_response(COULEURS_DISPONIBLES)

@app.route('/api/upload-model', methods=['POST'])
@timing_decorator
def upload_model():
    """Upload optimisé de modèle STAR"""
    try:
        if 'file' not in request.files:
            return create_error_response("Aucun fichier fourni")
        
        file = request.files['file']
        if file.filename == '':
            return create_error_response("Nom de fichier vide")
        
        if not allowed_file(file.filename):
            return create_error_response(f"Type de fichier non supporté. Extensions autorisées: {config.ALLOWED_EXTENSIONS}")
        
        # Sauvegarde sécurisée
        filename = secure_filename_custom(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Validation du fichier
        try:
            if filename.lower().endswith('.npz'):
                # Test de chargement
                test_data = np.load(filepath, allow_pickle=True)
                required_keys = ['v_template', 'f']
                missing_keys = [key for key in required_keys if key not in test_data.files]
                if missing_keys:
                    os.remove(filepath)
                    return create_error_response(f"Clés manquantes dans le fichier NPZ: {missing_keys}")
                test_data.close()
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return create_error_response(f"Fichier NPZ invalide: {str(e)}")
        
        file_info = {
            'filename': filename,
            'original_filename': file.filename,
            'filepath': filepath,
            'size_mb': round(os.path.getsize(filepath) / 1024 / 1024, 2),
            'upload_time': datetime.now().isoformat()
        }
        
        return create_success_response(file_info, "Modèle uploadé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'upload: {str(e)}")
        return create_error_response(f"Erreur lors de l'upload: {str(e)}")

@app.route('/api/generate-clothing', methods=['POST'])
@timing_decorator
@memory_monitor
def generate_clothing():
    """Génération optimisée de vêtement"""
    try:
        # Validation des données d'entrée
        data = request.get_json()
        if not data:
            return create_error_response("Données JSON manquantes")
        
        required_fields = ['model_file', 'measurements', 'clothing_type', 'color']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return create_error_response(f"Champs manquants: {missing_fields}")
        
        # Validation du fichier modèle
        model_filename = data['model_file']
        model_filepath = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
        if not os.path.exists(model_filepath):
            return create_error_response("Fichier modèle introuvable")
        
        # Validation des mesures
        measurements = validate_measurements(data['measurements'])
        if not measurements:
            return create_error_response("Aucune mesure valide fournie")
        
        # Validation du type de vêtement
        clothing_type = data['clothing_type']
        if clothing_type not in TYPES_VETEMENTS:
            return create_error_response(f"Type de vêtement non supporté: {clothing_type}")
        
        # Validation de la couleur
        color = data['color']
        if color not in COULEURS_DISPONIBLES:
            return create_error_response(f"Couleur non disponible: {color}")
        
        # Génération du vêtement
        with error_handler("Génération de vêtement"):
            result = generator.generer_vetement_complet_ultra_optimized(
                model_filepath, measurements, clothing_type, color
            )
        
        # Cache du résultat
        cache_key = f"clothing_{result['id']}"
        clothing_cache.set(cache_key, result)
        
        # Réponse optimisée (sans les données lourdes)
        response_data = {
            'id': result['id'],
            'type_vetement': result['type_vetement'],
            'couleur': result['couleur'],
            'mesures_finales': result['mesures_finales'],
            'metadata': result['metadata']
        }
        
        return create_success_response(response_data, "Vêtement généré avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {str(e)}")
        return create_error_response(f"Erreur lors de la génération: {str(e)}")

@app.route('/api/download-clothing/<clothing_id>', methods=['GET'])
@timing_decorator
def download_clothing(clothing_id: str):
    """Téléchargement optimisé de vêtement"""
    try:
        # Format de fichier
        format_fichier = request.args.get('format', 'obj').lower()
        if format_fichier not in ['obj', 'ply']:
            return create_error_response("Format non supporté. Utilisez 'obj' ou 'ply'")
        
        # Récupération depuis le cache
        cache_key = f"clothing_{clothing_id}"
        clothing_data = clothing_cache.get(cache_key)
        if not clothing_data:
            return create_error_response("Vêtement introuvable ou expiré")
        
        # Génération du fichier
        filename = f"vetement_{clothing_id}.{format_fichier}"
        filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
        
        success = generator.sauvegarder_fichier_optimized(
            clothing_data['vertices'],
            clothing_data['faces'],
            filepath,
            format_fichier
        )
        
        if not success:
            return create_error_response("Erreur lors de la génération du fichier")
        
        # Nettoyage automatique après téléchargement
        def cleanup():
            time.sleep(60)  # Attendre 1 minute
            if os.path.exists(filepath):
                os.remove(filepath)
        
        threading.Thread(target=cleanup, daemon=True).start()
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {str(e)}")
        return create_error_response(f"Erreur lors du téléchargement: {str(e)}")

@app.route('/api/clothing-preview/<clothing_id>', methods=['GET'])
@timing_decorator
def clothing_preview(clothing_id: str):
    """Prévisualisation optimisée de vêtement"""
    try:
        # Récupération depuis le cache
        cache_key = f"clothing_{clothing_id}"
        clothing_data = clothing_cache.get(cache_key)
        if not clothing_data:
            return create_error_response("Vêtement introuvable ou expiré")
        
        # Génération d'une image de prévisualisation
        try:
            mesh = generator.generer_mesh_vedo_optimized(
                clothing_data['vertices'],
                clothing_data['faces'],
                clothing_data['couleur_rgb']
            )
            
            # Configuration de la vue optimisée
            preview_file = os.path.join(config.TEMP_DIR, f"preview_{clothing_id}.png")
            
            # Rendu optimisé sans affichage
            mesh.show(
                interactive=False,
                offscreen=True,
                size=(800, 600),
                zoom=1.2,
                elevation=-30,
                azimuth=45
            ).screenshot(preview_file)
            
            # Lecture et encodage en base64
            with open(preview_file, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Nettoyage
            os.remove(preview_file)
            
            return create_success_response({
                'preview_image': f"data:image/png;base64,{img_data}",
                'clothing_info': {
                    'id': clothing_data['id'],
                    'type': clothing_data['type_vetement'],
                    'couleur': clothing_data['couleur']
                }
            })
            
        except Exception as e:
            logger.warning(f"Erreur lors de la génération de prévisualisation: {str(e)}")
            # Retourner les informations sans image
            return create_success_response({
                'preview_image': None,
                'clothing_info': {
                    'id': clothing_data['id'],
                    'type': clothing_data['type_vetement'],
                    'couleur': clothing_data['couleur']
                },
                'error': "Prévisualisation non disponible"
            })
        
    except Exception as e:
        logger.error(f"Erreur lors de la prévisualisation: {str(e)}")
        return create_error_response(f"Erreur lors de la prévisualisation: {str(e)}")

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Nettoyage optimisé du cache"""
    try:
        models_cache.clear()
        measurements_cache.clear()
        clothing_cache.clear()
        
        # Nettoyage des fichiers temporaires
        temp_files_deleted = 0
        for folder in [app.config['GENERATED_FOLDER'], config.TEMP_DIR]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    try:
                        os.remove(filepath)
                        temp_files_deleted += 1
                    except Exception:
                        continue
        
        return create_success_response({
            'caches_cleared': ['models', 'measurements', 'clothing'],
            'temp_files_deleted': temp_files_deleted
        }, "Cache nettoyé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage du cache: {str(e)}")
        return create_error_response(f"Erreur lors du nettoyage: {str(e)}")

# Gestionnaire d'erreurs global
@app.errorhandler(404)
def not_found_error(error):
    return create_error_response("Endpoint non trouvé", 404)

@app.errorhandler(405)
def method_not_allowed_error(error):
    return create_error_response("Méthode non autorisée", 405)

@app.errorhandler(413)
def request_entity_too_large_error(error):
    return create_error_response("Fichier trop volumineux", 413)

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Erreur serveur interne: {str(error)}")
    return create_error_response("Erreur serveur interne", 500)

# Nettoyage automatique périodique
def cleanup_periodic():
    """Nettoyage périodique des ressources"""
    while True:
        try:
            time.sleep(3600)  # Toutes les heures
            
            # Nettoyage des fichiers temporaires anciens
            cutoff_time = time.time() - 3600  # 1 heure
            for folder in [app.config['GENERATED_FOLDER'], config.TEMP_DIR]:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        filepath = os.path.join(folder, filename)
                        if os.path.getmtime(filepath) < cutoff_time:
                            try:
                                os.remove(filepath)
                            except Exception:
                                continue
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage périodique: {str(e)}")

# Démarrage du thread de nettoyage
cleanup_thread = threading.Thread(target=cleanup_periodic, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    logger.info("=== Démarrage de l'API STAR Clothing Generator Optimisée ===")
    logger.info(f"Configuration: {config}")
    logger.info(f"Types de vêtements disponibles: {len(TYPES_VETEMENTS)}")
    logger.info(f"Couleurs disponibles: {len(COULEURS_DISPONIBLES)}")
    
    # Configuration de production optimisée
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Désactivé pour la production
        threaded=True,
        use_reloader=False
    )
