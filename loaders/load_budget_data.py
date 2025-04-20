'''
import pandas as pd
from pathlib import Path

def load_powerplant_data(file_path):
    """
    Loads budget data for each power plant from a vertically stacked Excel sheet.
    Assumes each power plant is represented in 4 rows:
    - Row 0: Year header
    - Row 1: Boiler budget
    - Row 2: Turbine budget
    - Row 3: EIC budget
    """
    df_raw = pd.read_excel(file_path, header=None)

    plant_data = []
    plant_names = ['Plant A', 'Plant B', 'Plant C']  # Customize these names
    rows_per_plant = 4

    for i, plant in enumerate(plant_names):
        start_row = i * rows_per_plant
        table = df_raw.iloc[start_row:start_row + rows_per_plant]

        header = table.iloc[0, 1:].tolist()
        years = pd.to_datetime(header, format='%Y', errors='coerce').year

        for j, equipment in enumerate(['Boiler', 'Turbine', 'EIC']):
            values = table.iloc[j + 1, 1:].values
            plant_df = pd.DataFrame({
                'PowerPlant': plant,
                'Equipment': equipment,
                'Year': years,
                'Budget': values
            })
            plant_data.append(plant_df)

    combined_df = pd.concat(plant_data, ignore_index=True)
    return combined_df

# Example usage (can be called from app.py or a Jupyter notebook)
if __name__ == '__main__':
    file_path = Path("data/budget_data.xlsx")
    df = load_powerplant_data(file_path)
    print(df.head())
'''

# loaders/load_budget_data.py

import pandas as pd
from db.database import save_data, load_data

def load_budget_data_from_db() -> pd.DataFrame:
    """Return all records from the PostgreSQL table."""
    return load_data()

def load_budget_data_from_excel(uploaded_file) -> pd.DataFrame:
    """
    Read a long-format Excel file and save to DB.
    Expects columns: Plant, Equipment, Year, Budget, MaintenanceType
    """
    df = pd.read_excel(uploaded_file)
    required = {'Plant', 'Equipment', 'Year', 'Budget', 'MaintenanceType'}
    if not required.issubset(df.columns):
        raise ValueError(f"Excel must contain columns: {', '.join(required)}")
    save_data(df)
    return df
