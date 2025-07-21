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
from datetime import datetime
import base64
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'star_1_1')
MAPPING_PATH = os.path.join(BASE_DIR, 'star_measurements_mapping.json')
TEMP_DIR = tempfile.mkdtemp()
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated_clothes'
ALLOWED_EXTENSIONS = {'obj'}

# Création des dossiers
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Cache pour les modèles chargés
models_cache = {}
# _____________________________________________________fonctions pour génération de mesh 3D________________________________________________________________

# --- Fonctions utilitaires STAR ---
def charger_modele_star(npz_path):
    """Charge un modèle STAR depuis un fichier NPZ"""
    try:
        data = np.load(npz_path)
        v_template = data['v_template']
        f = data['f']
        J_regressor = data['J_regressor']
        shapedirs = data.get('shapedirs', None)
        posedirs = data.get('posedirs', None)
        Jtr = J_regressor.dot(v_template)
        
        return {
            'v_template': v_template,
            'f': f,
            'Jtr': Jtr,
            'J_regressor': J_regressor,
            'shapedirs': shapedirs,
            'posedirs': posedirs
        }
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du modèle: {str(e)}")

def charger_mapping():
    """Charge le mapping des mesures"""
    try:
        with open(MAPPING_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du mapping: {str(e)}")

def calculer_mesures_modele(vertices, joints, mapping):
    """Calcule les mesures actuelles du modèle"""
    mesures = {}
    for mesure, info in mapping.items():
        joint_indices = info["joints"]
        if len(joint_indices) == 2:
            # Distance entre deux joints
            mesures[mesure] = euclidean(joints[joint_indices[0]], joints[joint_indices[1]])
        elif len(joint_indices) == 1:
            # Pour les tours (approximation basée sur les vertices voisins)
            joint_pos = joints[joint_indices[0]]
            distances = np.linalg.norm(vertices - joint_pos, axis=1)
            nearby_vertices = vertices[distances < np.percentile(distances, 20)]
            if len(nearby_vertices) > 3:
                center = np.mean(nearby_vertices, axis=0)
                radii = np.linalg.norm(nearby_vertices - center, axis=1)
                mesures[mesure] = 2 * np.pi * np.mean(radii)
            else:
                mesures[mesure] = 50.0
    return mesures

def deformer_modele(v_template, shapedirs, mesures_cibles, mesures_actuelles, J_regressor):
    """Déforme le modèle pour s'approcher des mesures cibles"""
    if shapedirs is None:
        return v_template, np.zeros(10)
    
    n_betas = min(10, shapedirs.shape[2])
    
    def objective(betas):
        vertices_deformed = v_template + np.sum(shapedirs[:, :, :n_betas] * betas[None, None, :], axis=2)
        joints_deformed = J_regressor.dot(vertices_deformed)
        
        mesures_modele = {}
        for mesure in mesures_cibles.keys():
            if mesure in mesures_actuelles:
                ratio = mesures_cibles[mesure] / mesures_actuelles[mesure] if mesures_actuelles[mesure] > 0 else 1.0
                mesures_modele[mesure] = mesures_actuelles[mesure] * ratio
        
        error = 0
        for mesure in mesures_cibles.keys():
            if mesure in mesures_modele:
                error += (mesures_modele[mesure] - mesures_cibles[mesure]) ** 2
        
        regularization = 0.1 * np.sum(betas ** 2)
        return error + regularization
    
    initial_betas = np.zeros(n_betas)
    bounds = [(-3, 3)] * n_betas
    
    result = minimize(objective, initial_betas, method='L-BFGS-B', bounds=bounds)
    optimal_betas = result.x
    
    vertices_final = v_template + np.sum(shapedirs[:, :, :n_betas] * optimal_betas[None, None, :], axis=2)
    
    return vertices_final, optimal_betas

def get_model_info(model_data):
    """Récupère les informations sur le modèle"""
    info = {
        'vertices_count': len(model_data['v_template']),
        'faces_count': len(model_data['f']),
        'joints_count': len(model_data['Jtr']),
        'has_shapedirs': model_data['shapedirs'] is not None,
        'has_posedirs': model_data['posedirs'] is not None
    }
    
    if model_data['shapedirs'] is not None:
        info['shape_parameters_count'] = model_data['shapedirs'].shape[2]
    
    return info


# _________________________________________________________fonction_pour_generation_vetement_____________________________________________________________________

# --- COULEURS DISPONIBLES ---
COULEURS_DISPONIBLES = {
    "Noir": [25, 25, 25],
    "Bleu Marine": [25, 25, 128],
    "Gris Anthracite": [77, 77, 77],
    "Bordeaux": [128, 25, 51],
    "Rouge": [204, 25, 25],
    "Bleu Clair": [77, 102, 204],
    "Vert Olive": [102, 128, 51],
    "Marron": [102, 77, 51],
    "Blanc Cassé": [230, 230, 217],
    "Rose Poudré": [204, 153, 179],
}

# --- TYPES DE VETEMENTS ---
TYPES_VETEMENTS = {
    # Jupes droites
    "Mini-jupe droite": {
        "categorie": "jupe",
        "type": "droite",
        "longueur_relative": 0.15, 
        "description": "Mini-jupe droite (mi-cuisse)"
    },
    "Jupe droite au genou": {
        "categorie": "jupe",
        "type": "droite",
        "longueur_relative": 0.35, 
        "description": "Jupe droite classique (genou)"
    },
    "Jupe droite longue": {
        "categorie": "jupe",
        "type": "droite",
        "longueur_relative": 0.75, 
        "description": "Jupe droite longue (cheville)"
    },
    
    # Jupes ovales
    "Mini-jupe ovale": {
        "categorie": "jupe",
        "type": "ovale",
        "longueur_relative": 0.15,
        "ampleur": 1.3,
        "description": "Mini-jupe ovale évasée (mi-cuisse)"
    },
    "Jupe ovale au genou": {
        "categorie": "jupe",
        "type": "ovale",
        "longueur_relative": 0.35,
        "ampleur": 1.4,
        "description": "Jupe ovale classique (genou)"
    },
    "Jupe ovale longue": {
        "categorie": "jupe",
        "type": "ovale",
        "longueur_relative": 0.75,
        "ampleur": 1.5,
        "description": "Jupe ovale longue (cheville)"
    },
    "Jupe ovale bouffante": {
        "categorie": "jupe",
        "type": "ovale",
        "longueur_relative": 0.35,
        "ampleur": 1.8,
        "description": "Jupe ovale très évasée (style bouffant)"
    },
    
    # Jupes trapèze
    "Mini-jupe trapèze": {
        "categorie": "jupe",
        "type": "trapeze",
        "longueur_relative": 0.15,
        "evasement": 1.4,
        "description": "Mini-jupe trapèze évasée (mi-cuisse)"
    },
    "Jupe trapèze au genou": {
        "categorie": "jupe",
        "type": "trapeze",
        "longueur_relative": 0.35,
        "evasement": 1.6,
        "description": "Jupe trapèze classique (genou)"
    },
    "Jupe trapèze longue": {
        "categorie": "jupe",
        "type": "trapeze",
        "longueur_relative": 0.75,
        "evasement": 1.8,
        "description": "Jupe trapèze longue (cheville)"
    },
    "Jupe trapèze évasée": {
        "categorie": "jupe",
        "type": "trapeze",
        "longueur_relative": 0.45,
        "evasement": 2.0,
        "description": "Jupe trapèze très évasée (style A-line)"
    },
}

# --- FONCTIONS UTILITAIRES ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detecter_points_anatomiques(verts):
    """
    Détection avancée des points anatomiques avec gestion des bras
    """
    y_vals = verts[:, 1]
    x_vals = verts[:, 0]
    z_vals = verts[:, 2]
    
    # Points de base
    y_max = np.max(y_vals)
    y_min = np.min(y_vals)
    hauteur_totale = y_max - y_min
    
    # Points anatomiques détaillés
    y_tete = y_max - 0.1 * hauteur_totale
    y_epaules = y_max - 0.2 * hauteur_totale
    y_taille = y_max - 0.3 * hauteur_totale
    y_hanches = y_max - 0.45 * hauteur_totale
    y_genoux = y_max - 0.75 * hauteur_totale
    
    def calculer_rayon_a_hauteur(y_target, tolerance=0.05):
        mask = np.abs(y_vals - y_target) < tolerance
        if not np.any(mask):
            return 0.1
        points_niveau = verts[mask]
        distances = np.sqrt(points_niveau[:, 0]**2 + points_niveau[:, 2]**2)
        return np.percentile(distances, 75)
    
    # Rayons à différents niveaux
    rayon_tete = calculer_rayon_a_hauteur(y_tete)
    rayon_epaules = calculer_rayon_a_hauteur(y_epaules)
    rayon_taille = calculer_rayon_a_hauteur(y_taille)
    rayon_hanches = calculer_rayon_a_hauteur(y_hanches)
    
    # Détection des bras
    mask_zone_bras = (y_vals <= y_epaules) & (y_vals >= y_hanches)
    points_zone_bras = verts[mask_zone_bras]
    distances_radiales = np.sqrt(points_zone_bras[:, 0]**2 + points_zone_bras[:, 2]**2)
    seuil_bras = np.percentile(distances_radiales, 85)
    
    return {
        'y_tete': y_tete,
        'y_epaules': y_epaules,
        'y_taille': y_taille,
        'y_hanches': y_hanches,
        'y_genoux': y_genoux,
        'y_min': y_min,
        'y_max': y_max,
        'hauteur_totale': hauteur_totale,
        'rayon_tete': rayon_tete,
        'rayon_epaules': rayon_epaules,
        'rayon_taille': rayon_taille,
        'rayon_hanches': rayon_hanches,
        'seuil_bras': seuil_bras
    }

def calculer_profil_jupe_droite(points_anat, longueur_relative):
    """
    Calcule le profil d'une jupe droite avec la longueur spécifiée
    """
    y_taille = points_anat['y_taille']
    y_hanches = points_anat['y_hanches']
    y_min = points_anat['y_min']
    hauteur_totale = points_anat['hauteur_totale']
    
    rayon_taille = points_anat['rayon_taille']
    rayon_hanches = points_anat['rayon_hanches']
    
    y_debut_jupe = y_taille - 0.03
    y_bas_jupe = y_hanches - (longueur_relative * hauteur_totale)
    y_bas_jupe = max(y_bas_jupe, y_min + 0.1)
    
    rayon_debut = rayon_taille * 0.88
    rayon_hanches_jupe = rayon_hanches * 0.95
    rayon_bas = rayon_hanches_jupe * 1.02
    
    def rayon_a_hauteur(y):
        if y > y_debut_jupe:
            return 0
        elif y >= y_hanches:
            if y_debut_jupe == y_hanches:
                return rayon_debut
            t = (y_debut_jupe - y) / (y_debut_jupe - y_hanches)
            return rayon_debut + t * (rayon_hanches_jupe - rayon_debut)
        elif y >= y_bas_jupe:
            if y_hanches == y_bas_jupe:
                return rayon_hanches_jupe
            t = (y_hanches - y) / (y_hanches - y_bas_jupe)
            return rayon_hanches_jupe + t * (rayon_bas - rayon_hanches_jupe)
        else:
            return 0
    
    return {
        'y_debut': y_debut_jupe,
        'y_bas': y_bas_jupe,
        'rayon_fonction': rayon_a_hauteur,
        'rayon_debut': rayon_debut,
        'rayon_hanches': rayon_hanches_jupe,
        'rayon_bas': rayon_bas
    }

def calculer_profil_jupe_ovale(points_anat, longueur_relative, ampleur=1.4):
    """
    Calcule le profil d'une jupe ovale avec forme arrondie
    """
    y_taille = points_anat['y_taille']
    y_hanches = points_anat['y_hanches']
    y_min = points_anat['y_min']
    hauteur_totale = points_anat['hauteur_totale']
    
    rayon_taille = points_anat['rayon_taille']
    rayon_hanches = points_anat['rayon_hanches']
    
    y_debut_jupe = y_taille - 0.03
    y_bas_jupe = y_hanches - (longueur_relative * hauteur_totale)
    y_bas_jupe = max(y_bas_jupe, y_min + 0.1)
    
    rayon_debut = rayon_taille * 0.88
    rayon_max = rayon_hanches * ampleur
    rayon_bas = rayon_hanches * 0.9
    
    y_max_largeur = y_hanches - 0.1
    
    def rayon_a_hauteur(y):
        if y > y_debut_jupe:
            return 0
        elif y >= y_max_largeur:
            if y_debut_jupe == y_max_largeur:
                return rayon_debut
            t = (y_debut_jupe - y) / (y_debut_jupe - y_max_largeur)
            t_curve = 0.5 * (1 - np.cos(np.pi * t))
            return rayon_debut + t_curve * (rayon_max - rayon_debut)
        elif y >= y_bas_jupe:
            if y_max_largeur == y_bas_jupe:
                return rayon_max
            t = (y_max_largeur - y) / (y_max_largeur - y_bas_jupe)
            t_curve = 0.5 * (1 + np.cos(np.pi * t))
            return rayon_max - (rayon_max - rayon_bas) * (1 - t_curve)
        else:
            return 0
    
    return {
        'y_debut': y_debut_jupe,
        'y_bas': y_bas_jupe,
        'y_max_largeur': y_max_largeur,
        'rayon_fonction': rayon_a_hauteur,
        'rayon_debut': rayon_debut,
        'rayon_max': rayon_max,
        'rayon_bas': rayon_bas
    }

def calculer_profil_jupe_trapeze(points_anat, longueur_relative, evasement=1.6):
    """
    Calcule le profil d'une jupe trapèze avec évasement linéaire
    """
    y_taille = points_anat['y_taille']
    y_hanches = points_anat['y_hanches']
    y_min = points_anat['y_min']
    hauteur_totale = points_anat['hauteur_totale']
    
    rayon_taille = points_anat['rayon_taille']
    rayon_hanches = points_anat['rayon_hanches']
    
    y_debut_jupe = y_taille - 0.03
    y_bas_jupe = y_hanches - (longueur_relative * hauteur_totale)
    y_bas_jupe = max(y_bas_jupe, y_min + 0.1)
    
    rayon_debut = rayon_taille * 0.88
    rayon_bas = rayon_hanches * evasement
    
    def rayon_a_hauteur(y):
        if y > y_debut_jupe:
            return 0
        elif y >= y_bas_jupe:
            if y_debut_jupe == y_bas_jupe:
                return rayon_debut
            t = (y_debut_jupe - y) / (y_debut_jupe - y_bas_jupe)
            return rayon_debut + t * (rayon_bas - rayon_debut)
        else:
            return 0
    
    return {
        'y_debut': y_debut_jupe,
        'y_bas': y_bas_jupe,
        'rayon_fonction': rayon_a_hauteur,
        'rayon_debut': rayon_debut,
        'rayon_bas': rayon_bas
    }

def appliquer_forme_jupe(verts, profil_jupe):
    """
    Applique la forme de jupe aux vertices
    """
    verts_modifies = verts.copy()
    y_vals = verts[:, 1]
    
    masque_jupe = (y_vals <= profil_jupe['y_debut']) & (y_vals >= profil_jupe['y_bas'])
    
    for i, (x, y, z) in enumerate(verts):
        if masque_jupe[i]:
            distance_actuelle = np.sqrt(x**2 + z**2)
            
            if distance_actuelle > 0.001:
                nouveau_rayon = profil_jupe['rayon_fonction'](y)
                
                if nouveau_rayon > 0:
                    facteur = nouveau_rayon / distance_actuelle
                    verts_modifies[i, 0] = x * facteur
                    verts_modifies[i, 2] = z * facteur
    
    return verts_modifies, masque_jupe

def lisser_jupe(verts, masque_jupe, iterations=2):
    """
    Lisse la surface de la jupe
    """
    verts_lisses = verts.copy()
    
    for _ in range(iterations):
        nouveaux_verts = verts_lisses.copy()
        
        for i, est_jupe in enumerate(masque_jupe):
            if est_jupe:
                distances = np.linalg.norm(verts_lisses - verts_lisses[i], axis=1)
                voisins = np.where((distances < 0.03) & (distances > 0))[0]
                
                if len(voisins) > 2:
                    poids_centre = 0.75
                    poids_voisins = (1 - poids_centre) / len(voisins)
                    
                    nouveaux_verts[i] = (poids_centre * verts_lisses[i] + 
                                       poids_voisins * np.sum(verts_lisses[voisins], axis=0))
        
        verts_lisses = nouveaux_verts
    
    return verts_lisses

def creer_mesh_jupe_separe(verts_corps, masque_jupe, couleur_nom):
    """
    Crée un mesh séparé pour la jupe avec la couleur appropriée
    """
    points_jupe = verts_corps[masque_jupe]
    
    if len(points_jupe) == 0:
        logger.warning("Aucun point de jupe trouvé")
        return None
    
    faces = []
    indices_jupe = np.where(masque_jupe)[0]
    
    y_vals = points_jupe[:, 1]
    y_unique = np.unique(y_vals)
    
    for i in range(len(y_unique) - 1):
        y_actuel = y_unique[i]
        y_suivant = y_unique[i + 1]
        
        idx_actuel = np.where(np.abs(points_jupe[:, 1] - y_actuel) < 0.01)[0]
        idx_suivant = np.where(np.abs(points_jupe[:, 1] - y_suivant) < 0.01)[0]
        
        if len(idx_actuel) > 2 and len(idx_suivant) > 2:
            angles_actuel = np.arctan2(points_jupe[idx_actuel, 2], points_jupe[idx_actuel, 0])
            angles_suivant = np.arctan2(points_jupe[idx_suivant, 2], points_jupe[idx_suivant, 0])
            
            idx_actuel = idx_actuel[np.argsort(angles_actuel)]
            idx_suivant = idx_suivant[np.argsort(angles_suivant)]
            
            n_min = min(len(idx_actuel), len(idx_suivant))
            for j in range(n_min):
                k = (j + 1) % n_min
                faces.append([idx_actuel[j], idx_suivant[j], idx_actuel[k]])
                faces.append([idx_suivant[j], idx_suivant[k], idx_actuel[k]])
    
    try:
        mesh_jupe = Mesh([points_jupe, faces])
        couleur_rgb = COULEURS_DISPONIBLES[couleur_nom]
        mesh_jupe.color(couleur_rgb).alpha(0.9)
        return mesh_jupe
    except Exception as e:
        logger.warning(f"Erreur création mesh jupe: {e}")
        try:
            mesh_jupe = Mesh(points_jupe)
            couleur_rgb = COULEURS_DISPONIBLES[couleur_nom]
            mesh_jupe.color(couleur_rgb).alpha(0.9)
            return mesh_jupe
        except Exception as e2:
            logger.error(f"Impossible de créer le mesh jupe: {e2}")
            return None

def generer_vetement(mesh_corps, nom_vetement, couleur_nom):
    """
    Génère un vêtement selon les paramètres spécifiés
    """
    logger.info(f"Génération d'un(e) {nom_vetement.lower()}")
    
    verts = mesh_corps.points.copy()
    
    # Détecter les points anatomiques
    points_anat = detecter_points_anatomiques(verts)
    
    # Récupérer les paramètres du vêtement
    params_vetement = TYPES_VETEMENTS[nom_vetement]
    categorie = params_vetement["categorie"]
    type_vetement = params_vetement["type"]
    longueur_relative = params_vetement["longueur_relative"]
    
    # Calculer le profil selon le type de vêtement
    if categorie == "jupe":
        if type_vetement == "droite":
            profil_vetement = calculer_profil_jupe_droite(points_anat, longueur_relative)
        elif type_vetement == "ovale":
            ampleur = params_vetement.get("ampleur", 1.4)
            profil_vetement = calculer_profil_jupe_ovale(points_anat, longueur_relative, ampleur)
        elif type_vetement == "trapeze":
            evasement = params_vetement.get("evasement", 1.6)
            profil_vetement = calculer_profil_jupe_trapeze(points_anat, longueur_relative, evasement)
        
        # Appliquer la forme
        verts_modifies, masque_vetement = appliquer_forme_jupe(verts, profil_vetement)
        
        # Lisser
        verts_finaux = lisser_jupe(verts_modifies, masque_vetement, 1)
        
        # Créer le mesh
        mesh_vetement = creer_mesh_jupe_separe(verts_finaux, masque_vetement, couleur_nom)
    
    else:
        raise ValueError(f"Catégorie de vêtement non reconnue: {categorie}")
    
    # Modifier le mesh du corps
    mesh_corps.points = verts_finaux
    mesh_corps.color([255, 217, 179]).alpha(0.8)
    
    return mesh_corps, mesh_vetement

def sauvegarder_vetement(mesh_vetement, dossier, nom="vetement"):
    """Sauvegarde le mesh du vêtement"""
    if mesh_vetement is None:
        logger.error("Aucun mesh à sauvegarder !")
        return None
    
    os.makedirs(dossier, exist_ok=True)
    chemin = os.path.join(dossier, f"{nom}.obj")
    
    try:
        mesh_vetement.write(chemin)
        logger.info(f"Fichier sauvegardé : {chemin}")
        return chemin
    except Exception as e:
        logger.error(f"Impossible de sauvegarder : {e}")
        return None

# --- ROUTES API ---

# --- API ENDPOINTS ---

@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérification de l'état de l'API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/models/available', methods=['GET'])
def get_available_models():
    """Récupère la liste des modèles disponibles"""
    try:
        available_models = []
        for gender in ['male', 'female', 'neutral']:
            npz_path = os.path.join(BASE_DIR, gender, f"{gender}.npz")
            if os.path.exists(npz_path):
                available_models.append({
                    'gender': gender,
                    'path': npz_path,
                    'available': True
                })
            else:
                available_models.append({
                    'gender': gender,
                    'path': npz_path,
                    'available': False
                })
        
        return jsonify({
            'success': True,
            'models': available_models,
            'base_dir': BASE_DIR
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<gender>/load', methods=['POST'])
def load_model(gender):
    """Charge un modèle STAR spécifique"""
    try:
        if gender not in ['male', 'female', 'neutral']:
            return jsonify({
                'success': False,
                'error': 'Genre invalide. Utilisez: male, female, ou neutral'
            }), 400
        
        npz_path = os.path.join(BASE_DIR, gender, f"{gender}.npz")
        if not os.path.exists(npz_path):
            return jsonify({
                'success': False,
                'error': f'Modèle {gender} non trouvé à {npz_path}'
            }), 404
        
        # Charger le modèle
        model_data = charger_modele_star(npz_path)
        
        # Mettre en cache
        models_cache[gender] = model_data
        
        # Récupérer les informations du modèle
        model_info = get_model_info(model_data)
        
        return jsonify({
            'success': True,
            'gender': gender,
            'model_info': model_info,
            'message': f'Modèle {gender} chargé avec succès'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/measurements/mapping', methods=['GET'])
def get_measurements_mapping():
    """Récupère le mapping des mesures"""
    try:
        mapping = charger_mapping()
        return jsonify({
            'success': True,
            'mapping': mapping,
            'measurements_count': len(mapping)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<gender>/measurements', methods=['GET'])
def get_model_measurements(gender):
    """Calcule les mesures actuelles d'un modèle"""
    try:
        if gender not in models_cache:
            return jsonify({
                'success': False,
                'error': f'Modèle {gender} non chargé. Chargez-le d\'abord.'
            }), 400
        
        model_data = models_cache[gender]
        mapping = charger_mapping()
        
        # Calculer les mesures
        mesures = calculer_mesures_modele(
            model_data['v_template'], 
            model_data['Jtr'], 
            mapping
        )
        
        return jsonify({
            'success': True,
            'gender': gender,
            'measurements': mesures,
            'measurements_count': len(mesures)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<gender>/deform', methods=['POST'])
def deform_model(gender):
    """Déforme un modèle selon des mesures cibles"""
    try:
        if gender not in models_cache:
            return jsonify({
                'success': False,
                'error': f'Modèle {gender} non chargé. Chargez-le d\'abord.'
            }), 400
        
        data = request.get_json()
        if not data or 'target_measurements' not in data:
            return jsonify({
                'success': False,
                'error': 'Mesures cibles manquantes dans le body'
            }), 400
        
        target_measurements = data['target_measurements']
        
        # Valider les mesures
        if not isinstance(target_measurements, dict):
            return jsonify({
                'success': False,
                'error': 'Les mesures doivent être un dictionnaire'
            }), 400
        
        # Convertir les valeurs en float et valider
        try:
            validated_measurements = {}
            for key, value in target_measurements.items():
                float_value = float(value)
                if float_value <= 0:
                    return jsonify({
                        'success': False,
                        'error': f'Mesure {key} doit être positive'
                    }), 400
                validated_measurements[key] = float_value
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Valeur de mesure invalide: {str(e)}'
            }), 400
        
        model_data = models_cache[gender]
        mapping = charger_mapping()
        
        # Calculer les mesures actuelles
        current_measurements = calculer_mesures_modele(
            model_data['v_template'], 
            model_data['Jtr'], 
            mapping
        )
        
        # Déformer le modèle
        deformed_vertices, shape_params = deformer_modele(
            model_data['v_template'],
            model_data['shapedirs'],
            validated_measurements,
            current_measurements,
            model_data['J_regressor']
        )
        
        # Calculer les nouvelles mesures
        new_joints = model_data['J_regressor'].dot(deformed_vertices)
        new_measurements = calculer_mesures_modele(deformed_vertices, new_joints, mapping)
        
        # Sauvegarder le modèle déformé
        deformed_model_id = str(uuid.uuid4())
        models_cache[f"{gender}_deformed_{deformed_model_id}"] = {
            **model_data,
            'v_deformed': deformed_vertices,
            'shape_params': shape_params,
            'deformed_joints': new_joints,
            'target_measurements': validated_measurements,
            'achieved_measurements': new_measurements
        }
        
        return jsonify({
            'success': True,
            'gender': gender,
            'deformed_model_id': deformed_model_id,
            'shape_parameters': shape_params.tolist(),
            'target_measurements': validated_measurements,
            'achieved_measurements': new_measurements,
            'current_measurements': current_measurements,
            'message': 'Modèle déformé avec succès'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<gender>/export/obj', methods=['GET'])
def export_model_obj(gender):
    """Exporte un modèle en format OBJ"""
    try:
        # Vérifier si c'est un modèle déformé
        deformed_model_id = request.args.get('deformed_model_id')
        
        if deformed_model_id:
            model_key = f"{gender}_deformed_{deformed_model_id}"
            if model_key not in models_cache:
                return jsonify({
                    'success': False,
                    'error': 'Modèle déformé non trouvé'
                }), 404
            
            model_data = models_cache[model_key]
            vertices = model_data['v_deformed']
            filename = f"{gender}_deformed_{deformed_model_id}.obj"
        else:
            if gender not in models_cache:
                return jsonify({
                    'success': False,
                    'error': f'Modèle {gender} non chargé'
                }), 400
            
            model_data = models_cache[gender]
            vertices = model_data['v_template']
            filename = f"{gender}_template.obj"
        
        # Créer le contenu OBJ
        obj_content = "# Exported STAR model\n"
        obj_content += f"# Vertices: {len(vertices)}\n"
        obj_content += f"# Faces: {len(model_data['f'])}\n"
        
        if 'shape_params' in model_data:
            params_str = ', '.join(f'{p:.3f}' for p in model_data['shape_params'])
            obj_content += f"# Shape parameters: {params_str}\n"
        
        obj_content += "o STARMesh\n"
        
        # Vertices
        for v in vertices:
            obj_content += f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n"
        
        # Faces
        for face in model_data['f']:
            obj_content += f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n"
        
        # Créer un fichier temporaire
        temp_file = os.path.join(TEMP_DIR, filename)
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(obj_content)
        
        return send_file(temp_file, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<gender>/vertices', methods=['GET'])
def get_model_vertices(gender):
    """Récupère les vertices d'un modèle"""
    try:
        deformed_model_id = request.args.get('deformed_model_id')
        
        if deformed_model_id:
            model_key = f"{gender}_deformed_{deformed_model_id}"
            if model_key not in models_cache:
                return jsonify({
                    'success': False,
                    'error': 'Modèle déformé non trouvé'
                }), 404
            
            model_data = models_cache[model_key]
            vertices = model_data['v_deformed']
        else:
            if gender not in models_cache:
                return jsonify({
                    'success': False,
                    'error': f'Modèle {gender} non chargé'
                }), 400
            
            model_data = models_cache[gender]
            vertices = model_data['v_template']
        
        return jsonify({
            'success': True,
            'gender': gender,
            'vertices': vertices.tolist(),
            'faces': model_data['f'].tolist(),
            'vertices_count': len(vertices),
            'faces_count': len(model_data['f'])
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/<gender>/joints', methods=['GET'])
def get_model_joints(gender):
    """Récupère les joints d'un modèle"""
    try:
        deformed_model_id = request.args.get('deformed_model_id')
        
        if deformed_model_id:
            model_key = f"{gender}_deformed_{deformed_model_id}"
            if model_key not in models_cache:
                return jsonify({
                    'success': False,
                    'error': 'Modèle déformé non trouvé'
                }), 404
            
            model_data = models_cache[model_key]
            joints = model_data['deformed_joints']
        else:
            if gender not in models_cache:
                return jsonify({
                    'success': False,
                    'error': f'Modèle {gender} non chargé'
                }), 400
            
            model_data = models_cache[gender]
            joints = model_data['Jtr']
        
        return jsonify({
            'success': True,
            'gender': gender,
            'joints': joints.tolist(),
            'joints_count': len(joints)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/cache/clear', methods=['POST'])
def clear_cache():
    """Vide le cache des modèles"""
    try:
        cleared_models = list(models_cache.keys())
        models_cache.clear()
        
        return jsonify({
            'success': True,
            'message': 'Cache vidé avec succès',
            'cleared_models': cleared_models
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models/cache/status', methods=['GET'])
def get_cache_status():
    """Récupère l'état du cache"""
    try:
        cache_info = {}
        for key, model_data in models_cache.items():
            cache_info[key] = {
                'loaded': True,
                'vertices_count': len(model_data.get('v_template', [])) if 'v_template' in model_data else len(model_data.get('v_deformed', [])),
                'has_deformation': 'v_deformed' in model_data,
                'has_shape_params': 'shape_params' in model_data
            }
        
        return jsonify({
            'success': True,
            'cache_info': cache_info,
            'loaded_models_count': len(models_cache)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



# _________________________________________________________API pour la génération de vêtement_____________________________________________________________
# --- ROUTES API ---

@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérification de l'état de l'API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/clothing-types', methods=['GET'])
def get_clothing_types():
    """Récupère tous les types de vêtements disponibles"""
    try:
        types_organises = {
            "Jupes Droites": [],
            "Jupes Ovales": [],
            "Jupes Trapèze": []
        }
        
        for nom_vetement, params in TYPES_VETEMENTS.items():
            type_vetement = params["type"]
            if type_vetement == "droite":
                types_organises["Jupes Droites"].append({
                    "nom": nom_vetement,
                    "description": params["description"],
                    "longueur_relative": params["longueur_relative"]
                })
            elif type_vetement == "ovale":
                types_organises["Jupes Ovales"].append({
                    "nom": nom_vetement,
                    "description": params["description"],
                    "longueur_relative": params["longueur_relative"],
                    "ampleur": params.get("ampleur", 1.4)
                })
            elif type_vetement == "trapeze":
                types_organises["Jupes Trapèze"].append({
                    "nom": nom_vetement,
                    "description": params["description"],
                    "longueur_relative": params["longueur_relative"],
                    "evasement": params.get("evasement", 1.6)
                })
        
        return jsonify({
            'success': True,
            'data': types_organises,
            'total_types': len(TYPES_VETEMENTS)
        })
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des types: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/colors', methods=['GET'])
def get_colors():
    """Récupère toutes les couleurs disponibles"""
    try:
        colors_formatted = []
        for nom, rgb in COULEURS_DISPONIBLES.items():
            colors_formatted.append({
                'nom': nom,
                'rgb': rgb,
                'hex': f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            })
        
        return jsonify({
            'success': True,
            'data': colors_formatted,
            'total_colors': len(COULEURS_DISPONIBLES)
        })
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des couleurs: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-body', methods=['POST'])
def analyze_body():
    """Analyse un fichier .obj pour détecter les points anatomiques"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Aucun fichier sélectionné'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Charger le mesh
            mesh_corps = Mesh(filepath)
            verts = mesh_corps.points
            
            # Détecter les points anatomiques
            points_anat = detecter_points_anatomiques(verts)
            
            # Supprimer le fichier temporaire
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'data': {
                    'points_anatomiques': points_anat,
                    'nb_points': len(verts),
                    'dimensions': {
                        'hauteur': float(points_anat['hauteur_totale']),
                        'largeur_epaules': float(points_anat['rayon_epaules'] * 2),
                        'largeur_taille': float(points_anat['rayon_taille'] * 2),
                        'largeur_hanches': float(points_anat['rayon_hanches'] * 2)
                    }
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Format de fichier non supporté'}), 400
            
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du corps: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-clothing', methods=['POST'])
def generate_clothing():
    """Génère un vêtement sur un mannequin"""
    try:
        # Vérifier la présence du fichier
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Aucun fichier sélectionné'}), 400
        
        # Récupérer les paramètres
        nom_vetement = request.form.get('clothing_type')
        couleur_nom = request.form.get('color')
        
        if not nom_vetement or not couleur_nom:
            return jsonify({'success': False, 'error': 'Paramètres manquants'}), 400
        
        if nom_vetement not in TYPES_VETEMENTS:
            return jsonify({'success': False, 'error': 'Type de vêtement non reconnu'}), 400
        
        if couleur_nom not in COULEURS_DISPONIBLES:
            return jsonify({'success': False, 'error': 'Couleur non reconnue'}), 400
        
        if file and allowed_file(file.filename):
            # Sauvegarder le fichier temporairement
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Charger le mesh
            mesh_corps = Mesh(filepath)
            
            # Générer le vêtement
            mesh_corps_modifie, mesh_vetement = generer_vetement(mesh_corps, nom_vetement, couleur_nom)
            
            # Créer un nom de fichier unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nom_fichier = f"{nom_vetement.lower().replace(' ', '_').replace('-', '_')}_{couleur_nom.lower().replace(' ', '_')}_{timestamp}"
            
            # Sauvegarder le vêtement
            chemin_vetement = None
            if mesh_vetement:
                chemin_vetement = sauvegarder_vetement(mesh_vetement, app.config['GENERATED_FOLDER'], nom_fichier)
            
            # Sauvegarder le corps modifié
            chemin_corps = os.path.join(app.config['GENERATED_FOLDER'], f"corps_{nom_fichier}.obj")
            mesh_corps_modifie.write(chemin_corps)
            
            # Supprimer le fichier temporaire
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'data': {
                    'clothing_type': nom_vetement,
                    'color': couleur_nom,
                    'files': {
                        'clothing': chemin_vetement,
                        'body': chemin_corps
                    },
                    'parameters': TYPES_VETEMENTS[nom_vetement],
                    'color_rgb': COULEURS_DISPONIBLES[couleur_nom]
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Format de fichier non supporté'}), 400
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération du vêtement: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/preview-clothing', methods=['POST'])
def preview_clothing():
    """Génère un aperçu du vêtement sans sauvegarder"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        nom_vetement = request.form.get('clothing_type')
        couleur_nom = request.form.get('color')
        
        if not nom_vetement or not couleur_nom:
            return jsonify({'success': False, 'error': 'Paramètres manquants'}), 400
        
        if file and allowed_file(file.filename):
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_file:
                file.save(temp_file.name)
                
                # Charger et analyser le mesh
                mesh_corps = Mesh(temp_file.name)
                verts = mesh_corps.points
                
                # Détecter les points anatomiques
                points_anat = detecter_points_anatomiques(verts)
                
                # Calculer les dimensions du vêtement
                params_vetement = TYPES_VETEMENTS[nom_vetement]
                longueur_relative = params_vetement["longueur_relative"]
                
                # Calculer les dimensions prévues
                hauteur_vetement = longueur_relative * points_anat['hauteur_totale']
                y_bas_prevu = points_anat['y_hanches'] - hauteur_vetement
                
                # Supprimer le fichier temporaire
                os.unlink(temp_file.name)
                
                return jsonify({
                    'success': True,
                    'data': {
                        'clothing_type': nom_vetement,
                        'color': couleur_nom,
                        'dimensions': {
                            'hauteur_vetement': float(hauteur_vetement),
                            'position_debut': float(points_anat['y_taille']),
                            'position_fin': float(y_bas_prevu),
                            'rayon_taille': float(points_anat['rayon_taille']),
                            'rayon_hanches': float(points_anat['rayon_hanches'])
                        },
                        'parameters': params_vetement,
                        'color_rgb': COULEURS_DISPONIBLES[couleur_nom]
                    }
                })
        else:
            return jsonify({'success': False, 'error': 'Format de fichier non supporté'}), 400
            
    except Exception as e:
        logger.error(f"Erreur lors de l'aperçu: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Télécharge un fichier généré"""
    try:
        # Vérifier que le fichier existe dans le dossier des fichiers générés
        file_path = os.path.join(app.config['GENERATED_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'success': False, 'error': 'Fichier non trouvé'}), 404
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/list-generated', methods=['GET'])
def list_generated_files():
    """Liste tous les fichiers générés"""
    try:
        files = []
        if os.path.exists(app.config['GENERATED_FOLDER']):
            for filename in os.listdir(app.config['GENERATED_FOLDER']):
                if filename.endswith('.obj'):
                    filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
                    stat = os.stat(filepath)
                    files.append({
                        'filename': filename,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return jsonify({
            'success': True,
            'data': files,
            'total_files': len(files)
        })
    except Exception as e:
        logger.error(f"Erreur lors de la liste des fichiers: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/delete-generated/<filename>', methods=['DELETE'])
def delete_generated_file(filename):
    """Supprime un fichier généré"""
    try:
        file_path = os.path.join(app.config['GENERATED_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': 'Fichier supprimé avec succès'})
        else:
            return jsonify({'success': False, 'error': 'Fichier non trouvé'}), 404
    except Exception as e:
        logger.error(f"Erreur lors de la suppression: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/batch-generate', methods=['POST'])
def batch_generate():
    """Génère plusieurs vêtements en lot"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        batch_config = request.form.get('batch_config')
        
        if not batch_config:
            return jsonify({'success': False, 'error': 'Configuration du lot manquante'}), 400
        
        try:
            batch_config = json.loads(batch_config)
        except json.JSONDecodeError:
            return jsonify({'success': False, 'error': 'Configuration du lot invalide'}), 400
        
        if file and allowed_file(file.filename):
            # Sauvegarder le fichier temporairement
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            results = []
            
            for config in batch_config:
                try:
                    nom_vetement = config.get('clothing_type')
                    couleur_nom = config.get('color')
                    
                    if not nom_vetement or not couleur_nom:
                        results.append({
                            'config': config,
                            'success': False,
                            'error': 'Paramètres manquants'
                        })
                        continue
                    
                    if nom_vetement not in TYPES_VETEMENTS or couleur_nom not in COULEURS_DISPONIBLES:
                        results.append({
                            'config': config,
                            'success': False,
                            'error': 'Type de vêtement ou couleur non reconnu'
                        })
                        continue
                    
                    # Charger le mesh
                    mesh_corps = Mesh(filepath)
                    
                    # Générer le vêtement
                    mesh_corps_modifie, mesh_vetement = generer_vetement(mesh_corps, nom_vetement, couleur_nom)
                    
                    # Créer un nom de fichier unique
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nom_fichier = f"{nom_vetement.lower().replace(' ', '_').replace('-', '_')}_{couleur_nom.lower().replace(' ', '_')}_{timestamp}"
                    
                    # Sauvegarder
                    chemin_vetement = None
                    if mesh_vetement:
                        chemin_vetement = sauvegarder_vetement(mesh_vetement, app.config['GENERATED_FOLDER'], nom_fichier)
                    
                    chemin_corps = os.path.join(app.config['GENERATED_FOLDER'], f"corps_{nom_fichier}.obj")
                    mesh_corps_modifie.write(chemin_corps)
                    
                    results.append({
                        'config': config,
                        'success': True,
                        'files': {
                            'clothing': chemin_vetement,
                            'body': chemin_corps
                        }
                    })
                    
                except Exception as e:
                    results.append({
                        'config': config,
                        'success': False,
                        'error': str(e)
                    })
            
            # Supprimer le fichier temporaire
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'data': results,
                'total_processed': len(results),
                'successful': len([r for r in results if r['success']])
            })
        else:
            return jsonify({'success': False, 'error': 'Format de fichier non supporté'}), 400
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération en lot: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/validate-mesh', methods=['POST'])
def validate_mesh():
    """Valide un fichier mesh avant traitement"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Aucun fichier sélectionné'}), 400
        
        if file and allowed_file(file.filename):
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_file:
                file.save(temp_file.name)
                
                try:
                    # Charger le mesh
                    mesh_corps = Mesh(temp_file.name)
                    verts = mesh_corps.points
                    faces = mesh_corps.faces
                    
                    # Validations
                    validations = {
                        'has_vertices': len(verts) > 0,
                        'has_faces': len(faces) > 0,
                        'vertex_count': len(verts),
                        'face_count': len(faces),
                        'is_3d': verts.shape[1] == 3 if len(verts) > 0 else False,
                        'bbox': {
                            'min': verts.min(axis=0).tolist() if len(verts) > 0 else None,
                            'max': verts.max(axis=0).tolist() if len(verts) > 0 else None
                        }
                    }
                    
                    # Vérifications spécifiques pour le corps humain
                    if len(verts) > 0:
                        points_anat = detecter_points_anatomiques(verts)
                        validations['anatomical_points'] = points_anat
                        validations['height'] = float(points_anat['hauteur_totale'])
                        validations['is_human_like'] = points_anat['hauteur_totale'] > 1.0  # Hauteur réaliste
                    
                    # Supprimer le fichier temporaire
                    os.unlink(temp_file.name)
                    
                    is_valid = (validations['has_vertices'] and 
                               validations['has_faces'] and 
                               validations['is_3d'] and 
                               validations.get('is_human_like', False))
                    
                    return jsonify({
                        'success': True,
                        'is_valid': is_valid,
                        'validations': validations
                    })
                    
                except Exception as e:
                    os.unlink(temp_file.name)
                    return jsonify({
                        'success': False,
                        'is_valid': False,
                        'error': f'Erreur lors de la validation du mesh: {str(e)}'
                    })
        else:
            return jsonify({'success': False, 'error': 'Format de fichier non supporté'}), 400
            
    except Exception as e:
        logger.error(f"Erreur lors de la validation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-mesh-info', methods=['POST'])
def get_mesh_info():
    """Obtient des informations détaillées sur un mesh"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        if file and allowed_file(file.filename):
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_file:
                file.save(temp_file.name)
                
                try:
                    mesh_corps = Mesh(temp_file.name)
                    verts = mesh_corps.points
                    faces = mesh_corps.faces
                    
                    # Informations de base
                    info = {
                        'vertex_count': len(verts),
                        'face_count': len(faces),
                        'bbox': {
                            'min': verts.min(axis=0).tolist(),
                            'max': verts.max(axis=0).tolist(),
                            'size': (verts.max(axis=0) - verts.min(axis=0)).tolist()
                        }
                    }
                    
                    # Points anatomiques
                    points_anat = detecter_points_anatomiques(verts)
                    info['anatomical_points'] = points_anat
                    
                    # Dimensions corporelles
                    info['body_dimensions'] = {
                        'height': float(points_anat['hauteur_totale']),
                        'shoulder_width': float(points_anat['rayon_epaules'] * 2),
                        'waist_width': float(points_anat['rayon_taille'] * 2),
                        'hip_width': float(points_anat['rayon_hanches'] * 2)
                    }
                    
                    # Recommandations de vêtements
                    info['clothing_recommendations'] = []
                    for nom_vetement, params in TYPES_VETEMENTS.items():
                        longueur_relative = params['longueur_relative']
                        longueur_absolue = longueur_relative * points_anat['hauteur_totale']
                        
                        info['clothing_recommendations'].append({
                            'name': nom_vetement,
                            'description': params['description'],
                            'estimated_length': float(longueur_absolue),
                            'parameters': params
                        })
                    
                    # Supprimer le fichier temporaire
                    os.unlink(temp_file.name)
                    
                    return jsonify({
                        'success': True,
                        'data': info
                    })
                    
                except Exception as e:
                    os.unlink(temp_file.name)
                    return jsonify({'success': False, 'error': str(e)}), 500
        else:
            return jsonify({'success': False, 'error': 'Format de fichier non supporté'}), 400
            
    except Exception as e:
        logger.error(f"Erreur lors de l'obtention des informations: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Obtient des statistiques sur l'utilisation de l'API"""
    try:
        stats = {
            'total_clothing_types': len(TYPES_VETEMENTS),
            'total_colors': len(COULEURS_DISPONIBLES),
            'clothing_types_by_category': {},
            'generated_files': 0,
            'total_storage_size': 0
        }
        
        # Statistiques par catégorie
        for nom_vetement, params in TYPES_VETEMENTS.items():
            categorie = params['categorie']
            type_vetement = params['type']
            
            if categorie not in stats['clothing_types_by_category']:
                stats['clothing_types_by_category'][categorie] = {}
            
            if type_vetement not in stats['clothing_types_by_category'][categorie]:
                stats['clothing_types_by_category'][categorie][type_vetement] = 0
            
            stats['clothing_types_by_category'][categorie][type_vetement] += 1
        
        # Statistiques des fichiers générés
        if os.path.exists(app.config['GENERATED_FOLDER']):
            for filename in os.listdir(app.config['GENERATED_FOLDER']):
                if filename.endswith('.obj'):
                    stats['generated_files'] += 1
                    filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
                    stats['total_storage_size'] += os.path.getsize(filepath)
        
        return jsonify({
            'success': True,
            'data': stats
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de l'obtention des statistiques: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Vide le cache et les fichiers temporaires"""
    try:
        deleted_files = 0
        
        # Nettoyer le dossier des uploads
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    deleted_files += 1
        
        # Optionnel: nettoyer aussi les fichiers générés anciens (plus de 24h)
        if os.path.exists(app.config['GENERATED_FOLDER']):
            current_time = datetime.now().timestamp()
            for filename in os.listdir(app.config['GENERATED_FOLDER']):
                filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
                if os.path.isfile(filepath):
                    file_time = os.path.getmtime(filepath)
                    if current_time - file_time > 86400:  # 24 heures
                        os.remove(filepath)
                        deleted_files += 1
        
        return jsonify({
            'success': True,
            'message': f'{deleted_files} fichiers supprimés du cache'
        })
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage du cache: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500






# --- Endpoint pour les statistiques ---
@app.route('/api/stats', methods=['GET'])
def get_api_stats():
    """Récupère les statistiques de l'API"""
    try:
        stats = {
            'base_directory': BASE_DIR,
            'mapping_file': MAPPING_PATH,
            'temp_directory': TEMP_DIR,
            'loaded_models': len(models_cache),
            'available_endpoints': [
                'GET /api/health',
                'GET /api/models/available',
                'POST /api/models/<gender>/load',
                'GET /api/measurements/mapping',
                'GET /api/models/<gender>/measurements',
                'POST /api/models/<gender>/deform',
                'GET /api/models/<gender>/export/obj',
                'GET /api/models/<gender>/vertices',
                'GET /api/models/<gender>/joints',
                'POST /api/models/cache/clear',
                'GET /api/models/cache/status',
                'GET /api/stats'
            ]
        }
        
        return jsonify({
            'success': True,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# --- Gestion des erreurs ---
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint non trouvé',
        'message': 'Vérifiez l\'URL et la méthode HTTP'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Méthode HTTP non autorisée',
        'message': 'Vérifiez la méthode HTTP utilisée'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Erreur interne du serveur',
        'message': 'Une erreur inattendue s\'est produite'
    }), 500

if __name__ == '__main__':
    print("🚀 Démarrage de l'API STAR...")
    print(f"📁 Répertoire de base: {BASE_DIR}")
    print(f"📄 Fichier de mapping: {MAPPING_PATH}")
    print(f"🗂️ Répertoire temporaire: {TEMP_DIR}")
    print("🌐 API disponible sur: http://localhost:5000")
    print("📚 Documentation: http://localhost:5000/api/stats")
    
    # Créer les dossiers au démarrage
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(GENERATED_FOLDER, exist_ok=True)
    
    logger.info("Démarrage de l'API de génération de vêtements 3D")
    logger.info(f"Types de vêtements disponibles: {len(TYPES_VETEMENTS)}")
    logger.info(f"Couleurs disponibles: {len(COULEURS_DISPONIBLES)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)