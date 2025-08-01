import os
import json
import numpy as np
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
from PIL import Image
import tempfile
import atexit
import glob
import threading
import time

# Importations pour la visualisation 3D (EXACTEMENT comme dans vos scripts)
try:
    from vedo import Mesh, Plotter, show, colors as vcolors, Points
    VEDO_AVAILABLE = True
except ImportError:
    print("Vedo non disponible - visualisation 3D limitée")
    VEDO_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configuration des dossiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAR_DIR = os.path.join(BASE_DIR, 'star_1_1')
GENERATED_DIR = os.path.join(BASE_DIR, 'generated')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
PREVIEW_DIR = os.path.join(BASE_DIR, 'previews')

# Créer les dossiers nécessaires
for directory in [GENERATED_DIR, TEMP_DIR, PREVIEW_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- COULEURS DISPONIBLES (EXACTEMENT comme dans vos scripts) ---
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

# --- TYPES DE VÊTEMENTS (EXACTEMENT comme dans vos scripts) ---
TYPES_VETEMENTS = {
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
    "Jupe trapèze au genou": {
        "categorie": "jupe",
        "type": "trapeze",
        "longueur_relative": 0.35,
        "evasement": 1.6,
        "description": "Jupe trapèze classique (genou)"
    },
}

# --- MAPPING DES MESURES (comme dans le script 2) ---
DEFAULT_MAPPING = {
    "tour_poitrine": {"joints": [17, 18], "description": "Tour de poitrine"},
    "tour_taille": {"joints": [1], "description": "Tour de taille"},
    "tour_hanches": {"joints": [2], "description": "Tour de hanches"},
    "hauteur": {"joints": [0, 10], "description": "Hauteur totale"},
    "longueur_bras": {"joints": [16, 20], "description": "Longueur des bras"}
}

# ============== CLASSES EXACTES DE VOS SCRIPTS ==============

class MannequinGenerator:
    """EXACTEMENT la même logique que dans le script 2"""
    def __init__(self):
        self.v_template = None
        self.f = None
        self.Jtr = None
        self.J_regressor = None
        self.shapedirs = None
        self.posedirs = None
        
    def charger_modele_star(self, gender='neutral'):
        """COPIE EXACTE de la fonction du script 2"""
        npz_path = os.path.join(STAR_DIR, gender, f"{gender}.npz")
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Modèle STAR non trouvé: {npz_path}")
            
        data = np.load(npz_path)
        self.v_template = data['v_template']
        self.f = data['f']
        self.J_regressor = data['J_regressor']
        self.shapedirs = data.get('shapedirs', None)
        self.posedirs = data.get('posedirs', None)
        self.Jtr = self.J_regressor.dot(self.v_template)
        
        return True
    
    def calculer_mesures_modele(self, vertices, joints, mapping):
        """COPIE EXACTE du script 2"""
        mesures = {}
        for mesure, info in mapping.items():
            joint_indices = info["joints"]
            if len(joint_indices) == 2:
                mesures[mesure] = euclidean(joints[joint_indices[0]], joints[joint_indices[1]])
            elif len(joint_indices) == 1:
                # Pour les tours (approximation)
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
    
    def deformer_modele(self, mesures_cibles, mesures_actuelles):
        """COPIE EXACTE du script 2"""
        if self.shapedirs is None:
            print("Pas de blend shapes disponibles")
            return self.v_template, np.zeros(10)
        
        n_betas = min(10, self.shapedirs.shape[2])
        
        def objective(betas):
            vertices_deformed = self.v_template + np.sum(
                self.shapedirs[:, :, :n_betas] * betas[None, None, :], axis=2
            )
            joints_deformed = self.J_regressor.dot(vertices_deformed)
            
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
        betas = result.x
        
        vertices_final = self.v_template + np.sum(
            self.shapedirs[:, :, :n_betas] * betas[None, None, :], axis=2
        )
        
        return vertices_final, betas

class VetementGenerator:
    """EXACTEMENT les mêmes fonctions que dans le script 3"""
    
    @staticmethod
    def detecter_points_anatomiques(verts):
        """COPIE EXACTE du script 3"""
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
        
        rayon_tete = calculer_rayon_a_hauteur(y_tete)
        rayon_epaules = calculer_rayon_a_hauteur(y_epaules)
        rayon_taille = calculer_rayon_a_hauteur(y_taille)
        rayon_hanches = calculer_rayon_a_hauteur(y_hanches)
        
        distances_radiales = np.sqrt(verts[:, 0]**2 + verts[:, 2]**2)
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
    
    @staticmethod
    def calculer_profil_jupe_droite(points_anat, longueur_relative):
        """COPIE EXACTE du script 3"""
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
        }
    
    @staticmethod
    def calculer_profil_jupe_ovale(points_anat, longueur_relative, ampleur=1.4):
        """COPIE EXACTE du script 3"""
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
        }
    
    @staticmethod
    def calculer_profil_jupe_trapeze(points_anat, longueur_relative, evasement=1.6):
        """COPIE EXACTE du script 3"""
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
        }
    
    @staticmethod
    def appliquer_forme_jupe(verts, profil_jupe):
        """COPIE EXACTE du script 3"""
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
    
    @staticmethod
    def creer_mesh_jupe_separe(verts_corps, masque_jupe, couleur_nom):
        """VERSION CORRIGÉE pour éviter l'erreur .faces"""
        points_jupe = verts_corps[masque_jupe]
        
        if len(points_jupe) == 0:
            print("[WARN] Aucun point de jupe trouvé")
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
            if VEDO_AVAILABLE:
                mesh_jupe = Mesh([points_jupe, faces])
                couleur_rgb = COULEURS_DISPONIBLES[couleur_nom]
                # Normaliser les couleurs RGB (0-1) pour Vedo
                couleur_normalized = [c/255.0 for c in couleur_rgb]
                mesh_jupe.color(couleur_normalized).alpha(0.9)
                
                # CORRECTION: Retourner un dictionnaire avec les données nécessaires
                return {
                    'mesh_object': mesh_jupe,
                    'points': points_jupe,
                    'faces': faces,
                    'couleur': couleur_normalized
                }
            else:
                return None
        except Exception as e:
            print(f"[WARN] Erreur création mesh jupe: {e}")
            return None

# --- VISUALISATION 3D CORRIGÉE ---
class Visualisateur3D:
    def __init__(self):
        self.active_plotters = []
        
    def afficher_mannequin_avec_vetement(self, vertices_corps, faces, masque_vetement, 
                                       vertices_vetement, mesh_vetement_data, titre="Mannequin avec Vêtement"):
        """VERSION CORRIGÉE pour Vedo avec thread et fenêtrage correct"""
        if not VEDO_AVAILABLE:
            print("❌ Vedo non disponible")
            return False
        
        try:
            print(f"🎭 Affichage 3D: {titre}")
            
            # Créer le mesh du corps avec couleur peau (EXACTEMENT comme vos scripts)
            mesh_corps = Mesh([vertices_corps, faces])
            mesh_corps.color([0.8, 0.6, 0.4]).alpha(0.8)  # Couleur peau normalisée
            
            # Préparer la liste des meshes à afficher
            meshes_a_afficher = [mesh_corps]
            
            if mesh_vetement_data is not None and mesh_vetement_data['mesh_object'] is not None:
                meshes_a_afficher.append(mesh_vetement_data['mesh_object'])
                print(f"✅ Mesh vêtement ajouté")
            else:
                print("⚠️ Pas de mesh vêtement")
            
            # LANCEMENT VEDO avec gestion d'erreur spécifique
            def lancer_vedo():
                try:
                    # Créer un plotter explicite avec configuration
                    plt = Plotter(bg='white', axes=1, title=titre, interactive=True)
                    
                    # Ajouter tous les meshes
                    for mesh in meshes_a_affifier:
                        plt.add(mesh)
                    
                    # Configurer la vue (frontal + débout)
                    plt.camera.SetPosition(0, -3, 0)  # Position face au mannequin
                    plt.camera.SetFocalPoint(0, 0, 0)  # Regarder le centre
                    plt.camera.SetViewUp(0, 0, 1)     # Axe Z vers le haut
                    
                    # Afficher de manière interactive
                    plt.show(interactive=True)
                    plt.close()
                    
                except Exception as e:
                    print(f"❌ Erreur Vedo thread: {e}")
                    # Fallback: utilisation de show() simple
                    try:
                        show(*meshes_a_afficher, axes=1, viewup="z", bg="white", 
                             title=titre, interactive=True)
                    except Exception as e2:
                        print(f"❌ Erreur show() fallback: {e2}")
            
            # Lancer dans un thread séparé pour éviter le blocage
            thread = threading.Thread(target=lancer_vedo, daemon=True)
            thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur affichage avec vêtement: {e}")
            return False

# Function to clean up temporary files
def cleanup_temp_files():
    for directory in [TEMP_DIR, PREVIEW_DIR]:
        temp_files = glob.glob(os.path.join(directory, "*.*"))
        for file in temp_files:
            try:
                if os.path.getmtime(file) < time.time() - 3600:
                    os.remove(file)
            except:
                pass

atexit.register(cleanup_temp_files)

# Instance globale
mannequin_gen = MannequinGenerator()
visualisateur = Visualisateur3D()

# --- ROUTES API ---

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK',
        'message': 'API Mannequin et Vêtements CORRIGÉE',
        'star_models_available': os.path.exists(STAR_DIR),
        'couleurs_disponibles': len(COULEURS_DISPONIBLES),
        'types_vetements': len(TYPES_VETEMENTS),
        'vedo_available': VEDO_AVAILABLE,
        'corrections': [
            'Erreur Vedo .faces corrigée',
            'Vue de face miroir implémentée',
            'Visualisation 3D avec thread corrigée'
        ]
    })

@app.route('/api/vetement/generate', methods=['POST'])
def generate_vetement():
    """Route principale CORRIGÉE"""
    try:
        data = request.get_json()
        
        # Paramètres
        type_vetement = data.get('type_vetement', 'Jupe droite au genou')
        couleur = data.get('couleur', 'Bleu Marine')
        gender = data.get('gender', 'neutral')
        mesures = data.get('mesures', {})
        
        # Vérifications
        if type_vetement not in TYPES_VETEMENTS:
            return jsonify({'error': f'Type de vêtement non supporté: {type_vetement}'}), 400
        
        if couleur not in COULEURS_DISPONIBLES:
            return jsonify({'error': f'Couleur non disponible: {couleur}'}), 400
        
        print(f"👗 Génération vêtement CORRIGÉE: {type_vetement} {couleur}")
        
        # ÉTAPE 1: Charger le modèle STAR (comme script 2)
        mannequin_gen.charger_modele_star(gender)
        
        # ÉTAPE 2: Déformer si mesures fournies (comme script 2)
        if mesures:
            mesures_actuelles = mannequin_gen.calculer_mesures_modele(
                mannequin_gen.v_template, mannequin_gen.Jtr, DEFAULT_MAPPING
            )
            vertices_corps, betas = mannequin_gen.deformer_modele(mesures, mesures_actuelles)
        else:
            vertices_corps = mannequin_gen.v_template.copy()
            betas = np.zeros(10)
        
        # ÉTAPE 3: Générer le vêtement (EXACTEMENT comme script 3)
        params_vetement = TYPES_VETEMENTS[type_vetement]
        
        # Détecter les points anatomiques
        points_anat = VetementGenerator.detecter_points_anatomiques(vertices_corps)
        
        # Calculer le profil selon le type
        if params_vetement['type'] == 'droite':
            profil_vetement = VetementGenerator.calculer_profil_jupe_droite(
                points_anat, params_vetement['longueur_relative']
            )
        elif params_vetement['type'] == 'ovale':
            ampleur = params_vetement.get('ampleur', 1.4)
            profil_vetement = VetementGenerator.calculer_profil_jupe_ovale(
                points_anat, params_vetement['longueur_relative'], ampleur
            )
        elif params_vetement['type'] == 'trapeze':
            evasement = params_vetement.get('evasement', 1.6)
            profil_vetement = VetementGenerator.calculer_profil_jupe_trapeze(
                points_anat, params_vetement['longueur_relative'], evasement
            )
        
        # Appliquer la forme
        vertices_avec_vetement, masque_vetement = VetementGenerator.appliquer_forme_jupe(
            vertices_corps, profil_vetement
        )
        
        # CORRECTION: Créer le mesh du vêtement séparé (version corrigée)
        mesh_vetement_data = VetementGenerator.creer_mesh_jupe_separe(
            vertices_avec_vetement, masque_vetement, couleur
        )
        
        # ID unique
        vetement_id = f"vetement_{hash(str(data)) % 10000}"
        
        # CORRECTION: Sauvegarder pour visualisation
        temp_data = {
            'vertices_corps': vertices_corps,
            'vertices_avec_vetement': vertices_avec_vetement,
            'faces': mannequin_gen.f,
            'masque_vetement': masque_vetement,
            'mesh_vetement_data': {
                'points': mesh_vetement_data['points'] if mesh_vetement_data else None,
                'faces': mesh_vetement_data['faces'] if mesh_vetement_data else None,
                'couleur': mesh_vetement_data['couleur'] if mesh_vetement_data else None
            } if mesh_vetement_data else None,
            'couleur': couleur,
            'type_vetement': type_vetement,
            'gender': gender,
            'mesures': mesures
        }
        
        temp_file = os.path.join(TEMP_DIR, f"{vetement_id}.npz")
        np.savez(temp_file, **temp_data)
        
        nb_points_vetement = np.sum(masque_vetement)
        longueur_vetement = profil_vetement['y_debut'] - profil_vetement['y_bas']
        
        print(f"✅ Vêtement généré CORRIGÉ: {vetement_id}")
        print(f"   Points vêtement: {nb_points_vetement}")
        print(f"   Longueur: {longueur_vetement:.3f}")
        
        return jsonify({
            'success': True,
            'message': f'{type_vetement} {couleur.lower()} généré(e) - VERSION CORRIGÉE',
            'vetement_id': vetement_id,
            'info': {
                'type': type_vetement,
                'couleur': couleur,
                'gender': gender,
                'nb_points_vetement': int(nb_points_vetement),
                'longueur_vetement': float(longueur_vetement),
                'vertices_count': len(vertices_avec_vetement),
                'mesh_created': mesh_vetement_data is not None,
                'corrections_appliquees': True
            }
        })
        
    except Exception as e:
        print(f"❌ Erreur génération vêtement: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/vetement/preview/<vetement_id>', methods=['GET'])
def get_vetement_preview(vetement_id):
    """Génère un preview 2D VUE DE FACE MIROIR (comme un selfie)"""
    try:
        temp_file = os.path.join(TEMP_DIR, f"{vetement_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Vêtement non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices_corps = data['vertices_corps']
        vertices_avec_vetement = data['vertices_avec_vetement']
        faces = data['faces']
        masque_vetement = data['masque_vetement']
        couleur = str(data['couleur'])
        
        # CORRECTION: Générer le preview VUE DE FACE MIROIR
        img_buffer = generer_preview_face_miroir(
            vertices_corps, vertices_avec_vetement, faces, masque_vetement, couleur,
            width=600, height=800
        )
        
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=False,
            download_name=f'{vetement_id}_face_miroir.png'
        )
        
    except Exception as e:
        print(f"❌ Erreur preview vêtement: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/vetement/visualize/<vetement_id>', methods=['POST'])
def visualize_vetement_3d(vetement_id):
    """Lance la visualisation 3D CORRIGÉE"""
    try:
        if not VEDO_AVAILABLE:
            return jsonify({'error': 'Vedo non disponible pour la visualisation 3D'}), 400
        
        temp_file = os.path.join(TEMP_DIR, f"{vetement_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Vêtement non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices_corps = data['vertices_corps']
        vertices_avec_vetement = data['vertices_avec_vetement']
        faces = data['faces']
        masque_vetement = data['masque_vetement']
        couleur = str(data['couleur'])
        type_vetement = str(data['type_vetement'])
        
        print(f"🎭 Lancement visualisation 3D CORRIGÉE: {vetement_id}")
        
        # CORRECTION: Recréer le mesh du vêtement avec la version corrigée
        mesh_vetement_data = VetementGenerator.creer_mesh_jupe_separe(
            vertices_avec_vetement, masque_vetement, couleur
        )
        
        # CORRECTION: Lancer la visualisation dans un thread séparé avec gestion d'erreur
        def visualiser():
            try:
                titre = f"{type_vetement} {couleur} - Vue Corrigée"
                
                success = visualisateur.afficher_mannequin_avec_vetement(
                    vertices_corps,
                    faces,
                    masque_vetement,
                    vertices_avec_vetement,
                    mesh_vetement_data,
                    titre=titre
                )
                
                if success:
                    print(f"✅ Visualisation 3D CORRIGÉE réussie pour {vetement_id}")
                else:
                    print(f"❌ Échec visualisation 3D pour {vetement_id}")
                    
            except Exception as e:
                print(f"❌ Erreur thread visualisation: {e}")
        
        thread = threading.Thread(target=visualiser)
        thread.daemon = True
        thread.start()
        
        # Petite pause pour laisser le thread démarrer
        time.sleep(0.5)
        
        return jsonify({
            'success': True,
            'message': 'Visualisation 3D du vêtement lancée - VERSION CORRIGÉE',
            'vetement_id': vetement_id,
            'info': 'Fenêtre Vedo 3D interactive corrigée va s\'ouvrir',
            'corrections': [
                'Erreur .faces corrigée',
                'Thread Vedo amélioré',
                'Gestion d\'erreur ajoutée'
            ]
        })
        
    except Exception as e:
        print(f"❌ Erreur visualisation 3D: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mannequin/generate', methods=['POST'])
def generate_mannequin():
    """Génère un mannequin avec mesures personnalisées (comme script 2)"""
    try:
        data = request.get_json()
        
        gender = data.get('gender', 'neutral')
        mesures = data.get('mesures', {})
        
        print(f"🔧 Génération mannequin {gender}")
        
        # Charger le modèle (EXACTEMENT comme script 2)
        success = mannequin_gen.charger_modele_star(gender)
        if not success:
            return jsonify({'error': 'Impossible de charger le modèle'}), 500
        
        # Déformer selon les mesures (EXACTEMENT comme script 2)
        if mesures:
            mesures_actuelles = mannequin_gen.calculer_mesures_modele(
                mannequin_gen.v_template, mannequin_gen.Jtr, DEFAULT_MAPPING
            )
            vertices_final, betas = mannequin_gen.deformer_modele(mesures, mesures_actuelles)
            print(f"✅ Mesures appliquées: {mesures}")
        else:
            vertices_final = mannequin_gen.v_template.copy()
            betas = np.zeros(10)
            print(f"✅ Mannequin {gender} de base")
        
        # Informations sur le modèle généré
        info = {
            'gender': gender,
            'vertices_count': len(vertices_final),
            'faces_count': len(mannequin_gen.f),
            'joints_count': len(mannequin_gen.Jtr),
            'betas': betas.tolist(),
            'mesures_appliquees': mesures,
            'has_shapedirs': mannequin_gen.shapedirs is not None
        }
        
        # Sauvegarder temporairement
        mannequin_id = f"mannequin_{gender}_{hash(str(mesures)) % 10000}"
        temp_data = {
            'vertices': vertices_final,
            'faces': mannequin_gen.f,
            'joints': mannequin_gen.Jtr,
            'gender': gender,
            'mesures': mesures,
            'betas': betas
        }
        
        temp_file = os.path.join(TEMP_DIR, f"{mannequin_id}.npz")
        np.savez(temp_file, **temp_data)
        
        print(f"✅ Mannequin généré: {mannequin_id}")
        
        return jsonify({
            'success': True,
            'message': f'Mannequin {gender} généré',
            'info': info,
            'mannequin_id': mannequin_id
        })
        
    except Exception as e:
        print(f"❌ Erreur génération mannequin: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mannequin/preview/<mannequin_id>', methods=['GET'])
def get_mannequin_preview(mannequin_id):
    """Génère un preview PNG du mannequin VUE DE FACE MIROIR"""
    try:
        temp_file = os.path.join(TEMP_DIR, f"{mannequin_id}.npz")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Mannequin non trouvé'}), 404
        
        data = np.load(temp_file, allow_pickle=True)
        vertices = data['vertices']
        faces = data['faces']
        
        # CORRECTION: Générer preview de face miroir
        img_buffer = generer_preview_mannequin_face_miroir(vertices, faces)
        
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=False,
            download_name=f'{mannequin_id}_face_miroir.png'
        )
        
    except Exception as e:
        print(f"❌ Erreur preview: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/vetement/types', methods=['GET'])
def get_vetement_types():
    """Liste des types de vêtements disponibles"""
    types_detailles = {}
    
    for nom, params in TYPES_VETEMENTS.items():
        types_detailles[nom] = {
            'description': params['description'],
            'categorie': params['categorie'],
            'type': params['type'],
            'longueur_relative': params['longueur_relative']
        }
    
    return jsonify({
        'types_vetements': types_detailles,
        'couleurs': COULEURS_DISPONIBLES
    })

# --- FONCTIONS UTILITAIRES CORRIGÉES POUR LES PREVIEWS ---

def generer_preview_face_miroir(vertices_corps, vertices_vetement, faces, masque_vetement, couleur, width=600, height=800):
    """
    CORRECTION: Génère un preview VUE DE FACE MIROIR (comme un selfie)
    - Hauteur du mannequin parallèle à l'axe Y (ordonnées)
    - Vue de face comme si on regardait dans un miroir
    """
    
    fig = plt.figure(figsize=(width/100, height/100), dpi=100, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # CORRECTION: VUE DE FACE MIROIR
    # elevation=0 (vue horizontale), azimuth=0 (face)
    ax.view_init(elev=0, azim=0)
    ax.set_box_aspect([1,2,1])  # Ratio pour voir le mannequin debout
    
    # Afficher le corps (couleur peau)
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Corps en couleur peau
        triangles_corps = vertices_corps[faces]
        collection_corps = Poly3DCollection(triangles_corps, alpha=0.7, 
                                          facecolor='peachpuff', 
                                          edgecolor='lightgray',
                                          linewidth=0.1)
        ax.add_collection3d(collection_corps)
        
        # Vêtement avec la couleur spécifiée (VISIBLE et DISTINCTE)
        if np.any(masque_vetement):
            # Points du vêtement uniquement
            points_vetement = vertices_vetement[masque_vetement]
            
            # Couleur du vêtement (normalisée)
            couleur_rgb = COULEURS_DISPONIBLES[couleur]
            couleur_norm = [c/255.0 for c in couleur_rgb]
            
            # Afficher comme nuage de points colorés (plus visible)
            ax.scatter(points_vetement[:, 0], points_vetement[:, 1], points_vetement[:, 2], 
                      c=[couleur_norm], s=4, alpha=0.95, label=f'Vêtement {couleur}')
            
    except Exception as e:
        print(f"Erreur affichage 3D: {e}")
        # Fallback: nuage de points simple
        ax.scatter(vertices_vetement[:, 0], vertices_vetement[:, 1], vertices_vetement[:, 2], 
                  c='lightblue', s=1, alpha=0.7)
    
    # CORRECTION: Configuration pour vue de face miroir (mannequin debout)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # CORRECTION: Centrer sur le mannequin DEBOUT (Y = hauteur)
    center = np.mean(vertices_vetement, axis=0)
    
    # Calculer les bornes pour voir le mannequin ENTIER DEBOUT
    y_min, y_max = np.min(vertices_vetement[:, 1]), np.max(vertices_vetement[:, 1])  # Hauteur
    x_min, x_max = np.min(vertices_vetement[:, 0]), np.max(vertices_vetement[:, 0])  # Largeur
    z_min, z_max = np.min(vertices_vetement[:, 2]), np.max(vertices_vetement[:, 2])  # Profondeur
    
    # CORRECTION: Étendre pour voir tout le mannequin debout
    padding = 0.15
    ax.set_xlim(x_min - padding, x_max + padding)      # Largeur (gauche-droite)
    ax.set_ylim(y_min - padding, y_max + padding)      # Hauteur (pieds-tête)
    ax.set_zlim(z_min - padding, z_max + padding)      # Profondeur
    
    # CORRECTION: Fond blanc propre sans grilles
    fig.patch.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False  
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    # Titre
    plt.title(f'Vue Face Miroir - {couleur}', fontsize=12, pad=20)
    
    # Sauvegarder en mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    buf.seek(0)
    plt.close(fig)
    
    return buf

def generer_preview_mannequin_face_miroir(vertices, faces, width=600, height=800):
    """
    CORRECTION: Génère un preview du mannequin seul VUE DE FACE MIROIR
    - Mannequin debout (Y = hauteur)
    - Vue de face comme dans un miroir
    """
    
    fig = plt.figure(figsize=(width/100, height/100), dpi=100, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # CORRECTION: VUE DE FACE MIROIR pour mannequin debout
    ax.view_init(elev=0, azim=0)  # Face directe
    ax.set_box_aspect([1,2,1])    # Aspect pour mannequin debout
    
    # Afficher le mannequin
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        triangles = vertices[faces]
        collection = Poly3DCollection(triangles, alpha=0.8, 
                                    facecolor='peachpuff', 
                                    edgecolor='lightgray',
                                    linewidth=0.1)
        ax.add_collection3d(collection)
    except Exception as e:
        print(f"Erreur affichage mannequin: {e}")
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c='peachpuff', s=1, alpha=0.8)
    
    # Configuration axes
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # CORRECTION: Centrer le mannequin DEBOUT (Y = hauteur, pieds à tête)
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])  # Hauteur
    x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])  # Largeur  
    z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])  # Profondeur
    
    padding = 0.15
    ax.set_xlim(x_min - padding, x_max + padding)      # Largeur
    ax.set_ylim(y_min - padding, y_max + padding)      # Hauteur (pieds-tête)
    ax.set_zlim(z_min - padding, z_max + padding)      # Profondeur
    
    # Fond blanc propre
    fig.patch.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white') 
    ax.zaxis.pane.set_edgecolor('white')
    
    plt.title('Mannequin - Vue Face Miroir', fontsize=12, pad=20)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    buf.seek(0)
    plt.close(fig)
    
    return buf

if __name__ == '__main__':
    print("🚀 API Mannequin et Vêtements - VERSION CORRIGÉE")
    print(f"📁 Dossier STAR: {STAR_DIR}")
    print(f"📁 Dossier généré: {GENERATED_DIR}")
    print(f"🎨 Couleurs disponibles: {len(COULEURS_DISPONIBLES)}")
    print(f"👗 Types de vêtements: {len(TYPES_VETEMENTS)}")
    print(f"🎭 Vedo disponible: {VEDO_AVAILABLE}")
    print("✅ CORRECTIONS APPLIQUÉES:")
    print("   - Erreur 'Mesh.faces' corrigée")
    print("   - Vue de face miroir implémentée (Y=hauteur)")
    print("   - Visualisation 3D Vedo avec thread corrigé")
    print("   - Gestion d'erreur améliorée")
    print("💡 Preview 2D: Vue face miroir (comme selfie)")
    print("💡 Visualisation 3D: Fenêtre Vedo interactive corrigée")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )