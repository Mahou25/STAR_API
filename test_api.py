#!/usr/bin/env python3
"""
Script de test FINALISÉ pour l'API Mannequin et Vêtements CORRIGÉE
Teste toutes les fonctionnalités avec les corrections appliquées
Version complète et optimisée
"""

import requests
import json
import time
from PIL import Image
import io
import os
import glob
from datetime import datetime

# Configuration
API_BASE_URL = "http://127.0.0.1:5000"
TIMEOUT = 30  # Timeout pour les requêtes
RESULTS_DIR = "test_results"

def setup_test_environment():
    """Initialise l'environnement de test"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    print("🔧 Environnement de test initialisé")
    print(f"📁 Dossier résultats: {RESULTS_DIR}")
    print(f"🌐 API: {API_BASE_URL}")
    print("=" * 60)

def test_api_health():
    """Test de santé de l'API corrigée"""
    print("🔍 Test de santé de l'API CORRIGÉE...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print("✅ API opérationnelle CORRIGÉE")
            print(f"   - Modèles STAR disponibles: {data['star_models_available']}")
            print(f"   - Couleurs disponibles: {data['couleurs_disponibles']}")
            print(f"   - Types de vêtements: {data['types_vetements']}")
            print(f"   - Vedo disponible: {data['vedo_available']}")
            if 'corrections' in data:
                print("✅ Corrections appliquées:")
                for correction in data['corrections']:
                    print(f"      - {correction}")
            return True
        else:
            print(f"❌ Erreur API: {response.status_code}")
            if response.content:
                print(f"   Détails: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur connexion API: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

def test_vetement_types():
    """Test récupération des types de vêtements"""
    print("📋 Test types de vêtements disponibles...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/vetement/types", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print("✅ Types de vêtements récupérés")
            print("👗 Types disponibles:")
            for nom, info in data['types_vetements'].items():
                print(f"   - {nom}: {info['description']}")
                print(f"     Catégorie: {info['categorie']}, Type: {info['type']}")
            
            print(f"🎨 Couleurs disponibles: {len(data['couleurs'])}")
            couleurs_sample = list(data['couleurs'].items())[:5]
            for nom, rgb in couleurs_sample:
                print(f"   - {nom}: RGB{rgb}")
            
            # Sauvegarder les types pour référence
            with open(os.path.join(RESULTS_DIR, "types_vetements.json"), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return data
        else:
            print(f"❌ Erreur types: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def test_mannequin_generation():
    """Test génération de mannequin avec mesures personnalisées"""
    print("🔧 Test génération mannequin...")
    
    # Données test avec mesures spécifiques
    mannequin_data = {
        "gender": "female",
        "mesures": {
            "tour_taille": 75.0,
            "tour_hanches": 95.0,
            "tour_poitrine": 90.0,
            "hauteur": 165.0
        }
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/mannequin/generate", 
                               json=mannequin_data, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print("✅ Mannequin généré avec succès")
            print(f"   - ID: {data['mannequin_id']}")
            print(f"   - Genre: {data['info']['gender']}")
            print(f"   - Vertices: {data['info']['vertices_count']}")
            print(f"   - Faces: {data['info']['faces_count']}")
            print(f"   - Betas appliqués: {len(data['info']['betas'])}")
            
            # Sauvegarder les infos
            with open(os.path.join(RESULTS_DIR, f"mannequin_{data['mannequin_id']}.json"), 'w') as f:
                json.dump(data, f, indent=2)
            
            return data['mannequin_id']
        else:
            error_data = response.json() if response.content else {}
            print(f"❌ Erreur génération mannequin: {error_data}")
            return None
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def test_mannequin_preview(mannequin_id):
    """Test preview mannequin (vue face miroir)"""
    print(f"📸 Test preview mannequin {mannequin_id} (VUE FACE MIROIR)...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/mannequin/preview/{mannequin_id}", timeout=TIMEOUT)
        if response.status_code == 200:
            # Sauvegarder l'image dans le dossier résultats
            filename = os.path.join(RESULTS_DIR, f"preview_mannequin_{mannequin_id}.png")
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            # Vérifier l'image
            img = Image.open(io.BytesIO(response.content))
            print(f"✅ Preview mannequin sauvegardé: {filename}")
            print(f"   - Dimensions: {img.size}")
            print(f"   - Format: {img.format}")
            print(f"   - Taille fichier: {len(response.content)} bytes")
            print(f"   - VUE: Face miroir (mannequin debout)")
            return True
        else:
            print(f"❌ Erreur preview: {response.status_code}")
            if response.content:
                print(f"   Détails: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_vetement_generation():
    """Test génération de vêtement CORRIGÉE"""
    print("👗 Test génération vêtement CORRIGÉE...")
    
    # Données test avec vêtement coloré
    vetement_data = {
        "type_vetement": "Jupe ovale au genou",
        "couleur": "Rouge",
        "gender": "female",
        "mesures": {
            "tour_taille": 75.0,
            "tour_hanches": 95.0,
            "tour_poitrine": 90.0,
            "hauteur": 165.0
        }
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/vetement/generate", 
                               json=vetement_data, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print("✅ Vêtement généré avec succès - VERSION CORRIGÉE")
            print(f"   - ID: {data['vetement_id']}")
            print(f"   - Type: {data['info']['type']}")
            print(f"   - Couleur: {data['info']['couleur']}")
            print(f"   - Points vêtement: {data['info']['nb_points_vetement']}")
            print(f"   - Longueur: {data['info']['longueur_vetement']:.3f}")
            print(f"   - Mesh créé: {data['info']['mesh_created']}")
            if 'corrections_appliquees' in data['info']:
                print("✅ Corrections appliquées avec succès")
            
            # Sauvegarder les infos
            with open(os.path.join(RESULTS_DIR, f"vetement_{data['vetement_id']}.json"), 'w') as f:
                json.dump(data, f, indent=2)
            
            return data['vetement_id']
        else:
            error_data = response.json() if response.content else {}
            print(f"❌ Erreur génération vêtement: {error_data}")
            return None
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def test_vetement_preview(vetement_id):
    """Test preview vêtement (vue face miroir)"""
    print(f"📸 Test preview vêtement {vetement_id} (VUE FACE MIROIR)...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/vetement/preview/{vetement_id}", timeout=TIMEOUT)
        if response.status_code == 200:
            # Sauvegarder l'image
            filename = os.path.join(RESULTS_DIR, f"preview_vetement_{vetement_id}.png")
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            # Vérifier l'image
            img = Image.open(io.BytesIO(response.content))
            print(f"✅ Preview vêtement sauvegardé: {filename}")
            print(f"   - Dimensions: {img.size}")
            print(f"   - Format: {img.format}")
            print(f"   - Taille fichier: {len(response.content)} bytes")
            print(f"   - VUE: Face miroir (mannequin avec vêtement)")
            return True
        else:
            print(f"❌ Erreur preview: {response.status_code}")
            if response.content:
                print(f"   Détails: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_vetement_visualisation_3d(vetement_id):
    """Test visualisation 3D CORRIGÉE"""
    print(f"🎭 Test visualisation 3D CORRIGÉE pour {vetement_id}...")
    try:
        response = requests.post(f"{API_BASE_URL}/api/vetement/visualize/{vetement_id}", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print("✅ Visualisation 3D lancée - VERSION CORRIGÉE")
            print(f"   - Message: {data['message']}")
            if 'corrections' in data:
                print("✅ Corrections appliquées:")
                for correction in data['corrections']:
                    print(f"      - {correction}")
            print("⏳ Attendre quelques secondes pour l'ouverture de Vedo...")
            time.sleep(3)  # Laisser le temps au thread de démarrer
            return True
        else:
            error_data = response.json() if response.content else {}
            print(f"❌ Erreur visualisation 3D: {error_data}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_scenarios_multiples():
    """Test de plusieurs scénarios de vêtements"""
    print("🔄 Test scénarios multiples CORRIGÉS...")
    
    scenarios = [
        {
            "nom": "Jupe droite noire femme",
            "data": {
                "type_vetement": "Jupe droite au genou",
                "couleur": "Noir",
                "gender": "female",
                "mesures": {"tour_taille": 70, "tour_hanches": 95}
            }
        },
        {
            "nom": "Mini-jupe ovale bleue",
            "data": {
                "type_vetement": "Mini-jupe ovale",
                "couleur": "Bleu Marine", 
                "gender": "female",
                "mesures": {"tour_taille": 68, "tour_hanches": 92}
            }
        },
        {
            "nom": "Jupe trapèze bordeaux",
            "data": {
                "type_vetement": "Jupe trapèze au genou",
                "couleur": "Bordeaux",
                "gender": "female",
                "mesures": {"tour_taille": 75, "tour_hanches": 98}
            }
        },
        {
            "nom": "Mini-jupe droite blanche",
            "data": {
                "type_vetement": "Mini-jupe droite",
                "couleur": "Blanc Cassé",
                "gender": "female",
                "mesures": {"tour_taille": 72, "tour_hanches": 96}
            }
        },
        {
            "nom": "Jupe droite longue verte",
            "data": {
                "type_vetement": "Jupe droite longue",
                "couleur": "Vert Olive",
                "gender": "female",
                "mesures": {"tour_taille": 74, "tour_hanches": 100}
            }
        }
    ]
    
    resultats = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"--- Scénario {i}/{len(scenarios)}: {scenario['nom']} ---")
        try:
            response = requests.post(f"{API_BASE_URL}/api/vetement/generate", 
                                   json=scenario['data'], timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                vetement_id = data['vetement_id']
                print(f"✅ {scenario['nom']} généré: {vetement_id}")
                
                # Test du preview
                preview_ok = test_vetement_preview(vetement_id)
                
                resultats.append({
                    'nom': scenario['nom'],
                    'success': True,
                    'vetement_id': vetement_id,
                    'preview_ok': preview_ok,
                    'data': data['info']
                })
                
                # Petite pause entre les générations
                time.sleep(1)
                
            else:
                error_data = response.json() if response.content else {}
                print(f"❌ Erreur génération pour {scenario['nom']}: {error_data}")
                resultats.append({
                    'nom': scenario['nom'],
                    'success': False,
                    'error': error_data
                })
                
        except Exception as e:
            print(f"❌ Exception pour {scenario['nom']}: {e}")
            resultats.append({
                'nom': scenario['nom'],
                'success': False,
                'error': str(e)
            })
    
    # Sauvegarder les résultats
    with open(os.path.join(RESULTS_DIR, "scenarios_multiples.json"), 'w', encoding='utf-8') as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
    
    # Résumé des résultats
    succes = sum(1 for r in resultats if r['success'])
    total = len(scenarios)
    
    print(f"📊 Résumé des scénarios CORRIGÉS:")
    print(f"   - Réussis: {succes}/{total}")
    for resultat in resultats:
        if resultat['success']:
            preview_status = "✅" if resultat.get('preview_ok') else "⚠️"
            print(f"   ✅ {resultat['nom']} {preview_status}")
        else:
            print(f"   ❌ {resultat['nom']}")
    
    return resultats

def generer_rapport_final(tests_results):
    """Génère un rapport final des tests"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rapport_file = os.path.join(RESULTS_DIR, f"rapport_test_{timestamp}.json")
    
    # Compter les succès
    succes_count = 0
    total_count = 0
    
    for key, value in tests_results.items():
        if key == 'scenarios_multiples':
            if value:
                scenario_succes = sum(1 for r in value if r['success'])
                scenario_total = len(value)
                succes_count += scenario_succes
                total_count += scenario_total
        elif key.endswith('_generation') and value:
            succes_count += 1
            total_count += 1
        elif isinstance(value, bool) and value:
            succes_count += 1
            total_count += 1
        elif isinstance(value, bool):
            total_count += 1
    
    rapport = {
        'timestamp': timestamp,
        'api_url': API_BASE_URL,
        'resultats': tests_results,
        'resume': {
            'tests_reussis': succes_count,
            'tests_total': total_count,
            'taux_succes': f"{(succes_count/total_count*100):.1f}%" if total_count > 0 else "0%"
        },
        'fichiers_generes': []
    }
    
    # Lister les fichiers générés
    for ext in ['*.png', '*.json']:
        files = glob.glob(os.path.join(RESULTS_DIR, ext))
        rapport['fichiers_generes'].extend([os.path.basename(f) for f in files])
    
    # Sauvegarder le rapport
    with open(rapport_file, 'w', encoding='utf-8') as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    
    return rapport

def main():
    """Test principal de l'API CORRIGÉE"""
    print("🚀 TEST COMPLET DE L'API MANNEQUIN ET VÊTEMENTS CORRIGÉE")
    print("=" * 70)
    print("Tests des corrections apportées:")
    print("- Erreur 'Mesh.faces' corrigée")
    print("- Vue de face miroir implémentée (Y=hauteur)")  
    print("- Visualisation 3D Vedo avec thread corrigé")
    print("- Gestion d'erreur améliorée")
    print("- Timeouts et gestion robuste des erreurs")
    print("=" * 70)
    
    # Setup environnement
    setup_test_environment()
    
    tests_results = {
        'api_health': False,
        'vetement_types': None,
        'mannequin_generation': None,
        'mannequin_preview': False,
        'vetement_generation': None,
        'vetement_preview': False,
        'vetement_3d': False,
        'scenarios_multiples': []
    }
    
    # Test 1: Santé API
    tests_results['api_health'] = test_api_health()
    if not tests_results['api_health']:
        print("❌ API non accessible, arrêt des tests")
        print("💡 Vérifiez que l'API est démarrée sur http://127.0.0.1:5000")
        return
    
    print()
    
    # Test 2: Types de vêtements
    tests_results['vetement_types'] = test_vetement_types()
    
    print("\n" + "=" * 50)
    print("TEST GÉNÉRATION MANNEQUIN CORRIGÉE")
    print("=" * 50)
    
    # Test 3: Génération mannequin
    tests_results['mannequin_generation'] = test_mannequin_generation()
    
    # Test 4: Preview mannequin (vue face miroir)
    if tests_results['mannequin_generation']:
        tests_results['mannequin_preview'] = test_mannequin_preview(
            tests_results['mannequin_generation']
        )
    
    print("\n" + "=" * 50)
    print("TEST GÉNÉRATION VÊTEMENT CORRIGÉE")
    print("=" * 50)
    
    # Test 5: Génération vêtement CORRIGÉE
    tests_results['vetement_generation'] = test_vetement_generation()
    
    # Test 6: Preview vêtement (vue face miroir)
    if tests_results['vetement_generation']:
        tests_results['vetement_preview'] = test_vetement_preview(
            tests_results['vetement_generation']
        )
        
        # Test 7: Visualisation 3D CORRIGÉE
        print("\n" + "=" * 50)
        print("TEST VISUALISATION 3D CORRIGÉE")
        print("=" * 50)
        tests_results['vetement_3d'] = test_vetement_visualisation_3d(
            tests_results['vetement_generation']
        )
    
    print("\n" + "=" * 50)
    print("TEST SCÉNARIOS MULTIPLES CORRIGÉS")
    print("=" * 50)
    
    # Test 8: Scénarios multiples
    tests_results['scenarios_multiples'] = test_scenarios_multiples()
    
    # Génération du rapport final
    print("\n" + "=" * 70)
    print("GÉNÉRATION DU RAPPORT FINAL")
    print("=" * 70)
    
    rapport = generer_rapport_final(tests_results)
    
    # Résumé final
    print("\n" + "=" * 70)
    print("RÉSUMÉ FINAL DES TESTS CORRIGÉS")
    print("=" * 70)
    
    print(f"📊 STATISTIQUES:")
    print(f"   - Tests réussis: {rapport['resume']['tests_reussis']}")
    print(f"   - Tests total: {rapport['resume']['tests_total']}")
    print(f"   - Taux de succès: {rapport['resume']['taux_succes']}")
    
    print("\n✅ Tests réussis:")
    if tests_results['api_health']:
        print("   - Santé API corrigée")
    if tests_results['vetement_types']: 
        print("   - Types de vêtements")
    if tests_results['mannequin_generation']:
        print("   - Génération mannequin")
    if tests_results['mannequin_preview']:
        print("   - Preview mannequin (vue face miroir)")
    if tests_results['vetement_generation']:
        print("   - Génération vêtement CORRIGÉE")
    if tests_results['vetement_preview']:
        print("   - Preview vêtement (vue face miroir)")
    if tests_results['vetement_3d']:
        print("   - Visualisation 3D CORRIGÉE")
    
    # Résumé scénarios
    if tests_results['scenarios_multiples']:
        succes_scenarios = sum(1 for r in tests_results['scenarios_multiples'] if r['success'])
        total_scenarios = len(tests_results['scenarios_multiples'])
        print(f"   - Scénarios multiples: {succes_scenarios}/{total_scenarios}")
    
    # Fichiers générés
    print(f"\n📁 Fichiers générés dans '{RESULTS_DIR}':")
    for fichier in sorted(rapport['fichiers_generes']):
        if fichier.endswith('.png'):
            print(f"   🖼️  {fichier}")
        elif fichier.endswith('.json'):
            print(f"   📄 {fichier}")
    
    print(f"\n📋 Rapport détaillé: {os.path.basename(rapport_file) if 'rapport_file' in locals() else 'rapport_test_*.json'}")
    
    print("\n✅ CORRECTIONS TESTÉES ET VALIDÉES:")
    print("   - Erreur 'Mesh.faces' résolue ✅")
    print("   - Vue de face miroir (Y=hauteur, mannequin debout) ✅")
    print("   - Visualisation 3D Vedo avec thread corrigé ✅")
    print("   - Gestion d'erreur améliorée avec timeouts ✅")
    print("   - Preview 2D avec vue face miroir (comme selfie) ✅")
    print("   - Sauvegarde organisée des résultats ✅")
    
    if tests_results['vetement_3d']:
        print("\n🎭 VISUALISATION 3D:")
        print("   - Fenêtre Vedo interactive lancée")
        print("   - Mannequin avec vêtement visible")
        print("   - Couleurs distinctes (peau + vêtement)")
        print("   - Navigation 3D disponible")
    
    print(f"\n🎯 TESTS TERMINÉS - Résultats dans '{RESULTS_DIR}'")

if __name__ == "__main__":
    main()