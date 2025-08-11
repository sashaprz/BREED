import torch
import pickle
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from mp_api.client import MPRester
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

class OptimizedMaterialsProjectPretrainer:
    """Optimized pretraining system with major bottleneck fixes + caching"""

    def __init__(self, api_key: str, max_workers: int = 4, cache_dir: str = "materials_cache"):
        self.mpr = MPRester(api_key)
        self.max_workers = min(max_workers, 8)  
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print(f"> Initialized with {self.max_workers} workers, cache: {cache_dir}")

    def _get_cache_key(self, search_params: dict) -> str:
        param_str = json.dumps(search_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:12]

    def _save_materials_cache(self, materials: List[Dict], cache_key: str, metadata: dict = None):
        cache_data = {
            'materials': materials,
            'cached_at': datetime.now().isoformat(),
            'count': len(materials),
            'metadata': metadata or {}
        }
        cache_file = os.path.join(self.cache_dir, f"materials_{cache_key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"💾 Cached {len(materials)} materials -> {cache_file}")

    def _load_materials_cache(self, cache_key: str, max_age_hours: int = 168) -> Optional[List[Dict]]:
        cache_file = os.path.join(self.cache_dir, f"materials_{cache_key}.pkl")
        if not os.path.exists(cache_file):
            return None
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            cached_time = datetime.fromisoformat(cache_data['cached_at'])
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600
            if age_hours > max_age_hours:
                return None
            return cache_data['materials']
        except Exception:
            return None

    def fetch_electrolyte_materials_optimized(self, max_materials: int = 5000, use_cache: bool = True, force_refresh: bool = False):
        search_config = {
            'max_materials': max_materials,
            'searches': [
                (['Li'], "lithium compounds"),
                (['Na'], "sodium compounds"), 
                (['K'], "potassium compounds"),
                (['Mg'], "magnesium compounds"),
                (['Ca'], "calcium compounds"),
                (['Al'], "aluminum compounds"),
                (['Si'], "silicon compounds"),
                (['P'], "phosphorus compounds"),
                (['S'], "sulfur compounds"),
                (['F'], "fluoride compounds")
            ],
            'properties': [
                'material_id', 'formula_pretty', 'structure',
                'band_gap', 'formation_energy_per_atom',
                'energy_above_hull', 'density'
            ]
        }

        cache_key = self._get_cache_key(search_config)
        if use_cache and not force_refresh:
            cached_materials = self._load_materials_cache(cache_key)
            if cached_materials is not None:
                print(f"📂 Loaded materials from cache ({len(cached_materials)})")
                return cached_materials[:max_materials]

        print("> Fetching from Materials Project API...")
        all_materials = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._fetch_single_search, 
                    elements, search_config['properties'], description,
                    max_materials // len(search_config['searches']) + 100
                ): description for elements, description in search_config['searches']
            }
            for future in as_completed(futures):
                desc = futures[future]
                try:
                    materials = future.result(timeout=30)
                    all_materials.extend(materials)
                    print(f"  ✓ {desc}: {len(materials)} entries")
                except Exception as e:
                    print(f"  ✗ {desc} failed: {e}")

        unique_materials = {mat['material_id']: mat for mat in all_materials}
        result = list(unique_materials.values())[:max_materials]
        print(f"> Total unique materials: {len(result)}")

        if use_cache:
            self._save_materials_cache(result, cache_key, {'search_type': 'electrolyte_materials'})
        return result

    def _fetch_single_search(self, elements, properties, description, chunk_size):
        try:
            docs = self.mpr.materials.summary.search(
                elements=elements,
                fields=properties,
                chunk_size=min(chunk_size, 500)
            )
            return self._process_docs(docs, max_count=chunk_size)
        except:
            return []

    def _process_docs(self, docs, max_count=None):
        materials = []
        for i, doc in enumerate(docs):
            if max_count and i >= max_count:
                break
            if hasattr(doc, 'deprecated') and doc.deprecated:
                continue
            if not hasattr(doc, 'structure') or doc.structure is None:
                continue
            entry = {
                "material_id": str(doc.material_id),
                "formula": str(doc.formula_pretty) if doc.formula_pretty else "Unknown",
                "structure": doc.structure,
                **{
                    field: float(getattr(doc, field)) if getattr(doc, field) is not None else None
                    for field in ['band_gap', 'formation_energy_per_atom', 'energy_above_hull', 'density']
                }
            }
            materials.append(entry)
        return materials


if __name__ == "__main__":
    API_KEY = "tQ53EaqRe8UndenrzdDrDcg3vZypqn0d"  # Replace with your actual Materials Project API key
    pretrainer = OptimizedMaterialsProjectPretrainer(API_KEY, max_workers=4, cache_dir="materials_cache")

    # Scrape and save only
    materials_data = pretrainer.fetch_electrolyte_materials_optimized(
        max_materials=1000, use_cache=True, force_refresh=False
    )

    save_path = "scraped_materials.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(materials_data, f)

    print(f"✅ Saved {len(materials_data)} materials to {save_path}")
    print("Exiting now — no training performed.")
