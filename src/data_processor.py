"""
Data processor class for fetching professor data from PlanetTerp API
"""

import planetterp
from typing import List, Dict, Optional
from utils.helpers import log_progress
from config.config import PROFESSORS_PER_BATCH, PROGRESS_INTERVAL


class PlanetTerpDataProcessor:
    """Handles data fetching from PlanetTerp API"""
    
    def __init__(self):
        """Initialize the data processor"""
        pass
    
    def fetch_professor_data(self, professor_id: Optional[str] = None, 
                           max_professors: int = 1000) -> List[Dict]:
        """
        Fetch data from the PlanetTerp API using the Python wrapper
        
        Args:
            professor_id: Specific professor to fetch (optional)
            max_professors: Maximum number of professors to fetch
            
        Returns:
            List of professor dictionaries with reviews
        """
        professors = []

        # Fetch specific professor if requested
        if professor_id:
            try:
                professor = planetterp.professor(name=professor_id, reviews="true")
                professors = [professor] if professor else []
            except Exception as e:
                print(f"Error fetching professor {professor_id}: {e}")
                professors = []
        else:
            # Fetch multiple professors
            try:
                remaining = max_professors
                all_professors = []
                batch = 1
                
                while remaining > 0:
                    batch_size = min(PROFESSORS_PER_BATCH, remaining)
                    offset = (batch - 1) * PROFESSORS_PER_BATCH

                    print(f"Fetching batch {batch} (limit={batch_size}, offset={offset})")

                    try:
                        profs_batch = planetterp.professors(limit=batch_size, offset=offset)

                        if isinstance(profs_batch, list) and len(profs_batch) > 0:
                            valid_profs = [p for p in profs_batch if isinstance(p, dict)]
                            all_professors.extend(valid_profs)
                            print(f"Fetched {len(valid_profs)} professors in batch {batch}")

                            if len(profs_batch) < batch_size:
                                print("Reached the end of available professors")
                                break
                        else:
                            print(f"No more professors to fetch or invalid response in batch {batch}")
                            break

                    except Exception as e:
                        print(f"Error fetching professors batch {batch}: {e}")
                        break

                    remaining -= batch_size
                    batch += 1

                professors = all_professors[:max_professors]
                print(f"Successfully fetched {len(professors)} professors in total")

                # Fetch reviews for each professor
                self._fetch_reviews_for_professors(professors)
                
            except Exception as e:
                print(f"Error in professor fetching process: {e}")

        return professors
    
    def _fetch_reviews_for_professors(self, professors: List[Dict]) -> None:
        """
        Fetch reviews for each professor in the list
        
        Args:
            professors: List of professor dictionaries to update with reviews
        """
        print("Fetching reviews for each professor")
        total = len(professors)

        for i, professor in enumerate(professors):
            if isinstance(professor, dict) and professor.get('name'):
                try:
                    log_progress(i + 1, total, PROGRESS_INTERVAL, "Fetching reviews")

                    prof_with_reviews = planetterp.professor(name=professor['name'], reviews="true")

                    if isinstance(prof_with_reviews, dict):
                        professor['reviews'] = prof_with_reviews.get('reviews', [])
                    else:
                        professor['reviews'] = []

                except Exception as e:
                    print(f"Error fetching reviews for {professor['name']}: {e}")
                    professor['reviews'] = []