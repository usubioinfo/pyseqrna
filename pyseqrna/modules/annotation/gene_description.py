"""
Gene Description Service Module

This module provides functionality to fetch gene names and descriptions
from Ensembl BioMart for both plants and animals.

Features:
    - Fetch gene names and descriptions from Ensembl BioMart
    - Support for both plants and animals organism types
    - XML-based querying of BioMart web services
    - Caching of fetched descriptions to optimize performance
    - Integration with pandas DataFrames for annotating tables of genes

Configuration:
    The service relies on parameters:
    - species identifier (e.g., 'athaliana')
    - organism_type ('plants' or 'animals')

Dependencies:
    - requests
    - pandas

Classes:
    GeneDescriptionService - Service for fetching gene names and descriptions from BioMart.
    GeneAnnotator - Utility class for adding gene descriptions to DataFrames.

:Created: May 20, 2021
:Updated: March 10, 2026
:Author: Naveen Duhan
:Version: 1.0.0
"""

import requests
import pandas as pd
from io import StringIO
from xml.etree import ElementTree
from future.utils import native_str
import logging
from typing import Dict, List

HTTP_TIMEOUT_SECONDS = 60


class GeneDescriptionService:
    """
    Service for fetching gene names and descriptions from BioMart.

    This class handles the communication with Ensembl BioMart to retrieve
    gene annotations including gene names and descriptions.
    """

    def __init__(self, species: str, organism_type: str = "plants"):
        """
        Initialize the gene description service.

        Args:
            species: Species identifier (e.g., 'athaliana' for Arabidopsis thaliana)
            organism_type: Type of organism - 'plants' or 'animals'
        """
        self.species = species
        self.organism_type = organism_type.lower()
        self.logger = logging.getLogger(__name__)

        # Cache for gene descriptions to avoid repeated API calls
        self._gene_descriptions = None

        # Validate organism type
        if self.organism_type not in ["plants", "animals"]:
            raise ValueError("organism_type must be 'plants' or 'animals'")

    def _get_request(self, url: str, **params) -> requests.Response:
        """
        Make HTTP request to BioMart.

        Args:
            url: URL to request
            **params: Additional parameters for the request

        Returns:
            Response object

        Raises:
            requests.HTTPError: If the request fails
        """
        if params:
            response = requests.get(url, params=params, stream=True, timeout=HTTP_TIMEOUT_SECONDS)
        else:
            response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)

        response.raise_for_status()
        return response

    def _add_attr_node(self, root: ElementTree.Element, attr: str) -> None:
        """
        Add attribute node to XML query.

        Args:
            root: XML root element
            attr: Attribute name to add
        """
        attr_el = ElementTree.SubElement(root, "Attribute")
        attr_el.set("name", attr)

    def _build_biomart_query(self) -> str:
        """
        Build XML query for BioMart.

        Returns:
            XML query string for BioMart
        """
        # Determine BioMart configuration based on organism type
        if self.organism_type == "animals":
            scheme = "default"
            species_name = f"{self.species}_gene_ensembl"
        elif self.organism_type == "plants":
            scheme = "plants_mart"
            species_name = f"{self.species}_eg_gene"

        # Build XML query
        root = ElementTree.Element("Query")
        root.set("virtualSchemaName", scheme)
        root.set("formatter", "TSV")
        root.set("header", "1")
        root.set("uniqueRows", native_str(int(True)))
        root.set("datasetConfigVersion", "0.6")

        dataset = ElementTree.SubElement(root, "Dataset")
        dataset.set("name", species_name)
        dataset.set("interface", "default")

        # Add attributes we want to retrieve
        attributes = ["ensembl_gene_id", "external_gene_name", "description"]
        for attr in attributes:
            self._add_attr_node(dataset, attr)

        return ElementTree.tostring(root)

    def fetch_gene_descriptions(self) -> pd.DataFrame:
        """
        Fetch gene descriptions from BioMart.

        Returns:
            DataFrame with columns: Gene, Name, Description

        Raises:
            requests.HTTPError: If BioMart request fails
            Exception: If parsing fails
        """
        if self._gene_descriptions is not None:
            return self._gene_descriptions

        try:
            # Sanitize species name for logging to prevent log injection
            safe_species = str(self.species).replace("\n", "").replace("\r", "")
            self.logger.info(f"Fetching gene descriptions for {safe_species} from BioMart")

            # Determine URI based on organism type
            if self.organism_type == "animals":
                uri = "https://ensembl.org/biomart/martservice"
            else:  # plants
                uri = "https://plants.ensembl.org/biomart/martservice"

            # Build and execute query
            query = self._build_biomart_query()
            response = self._get_request(uri, query=query)

            # Parse response
            result = pd.read_csv(StringIO(response.text), sep="\t")
            result.columns = ["Gene", "Name", "Description"]

            # Cache the results
            self._gene_descriptions = result

            safe_count = str(len(result)).replace("\n", "").replace("\r", "")
            self.logger.info(f"Successfully fetched {safe_count} gene descriptions")
            return result

        except Exception as e:
            # Sanitize error message for logging to prevent log injection
            safe_error = str(e).replace("\n", "").replace("\r", "")
            self.logger.error(f"Failed to fetch gene descriptions: {safe_error}")
            raise

    def get_gene_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Get gene ID to name/description mapping as dictionary.

        Returns:
            Dictionary mapping gene IDs to {'name': name, 'description': description}
        """
        descriptions = self.fetch_gene_descriptions()

        mapping = {}
        for _, row in descriptions.iterrows():
            mapping[row["Gene"]] = {
                "name": row["Name"],
                "description": row["Description"],
            }

        return mapping


class GeneAnnotator:
    """
    Utility class for adding gene descriptions to DataFrames.

    This class provides methods to add gene names and descriptions
    to existing DataFrames containing gene lists.
    """

    def __init__(self, gene_description_service: GeneDescriptionService):
        """
        Initialize the gene annotator.

        Args:
            gene_description_service: Service for fetching gene descriptions
        """
        self.gene_service = gene_description_service
        self.logger = logging.getLogger(__name__)

    def add_descriptions_to_dataframe(
        self, df: pd.DataFrame, gene_column: str = "Gene", insert_position: int = 1
    ) -> pd.DataFrame:
        """
        Add gene names and descriptions to a DataFrame.

        Args:
            df: DataFrame containing genes
            gene_column: Name of the column containing gene IDs
            insert_position: Position where to insert Name and Description columns

        Returns:
            DataFrame with added Name and Description columns
        """
        try:
            self.logger.info("Adding gene descriptions to DataFrame")

            # Get gene descriptions
            gene_descriptions = self.gene_service.fetch_gene_descriptions()

            # Create a copy of the original DataFrame
            result_df = df.copy()

            # Merge with gene descriptions
            result_df = result_df.merge(gene_descriptions, left_on=gene_column, right_on="Gene", how="left")

            # Reorder columns to insert Name and Description at desired position
            columns = list(result_df.columns)

            # Remove Name and Description from their current positions
            if "Name" in columns:
                columns.remove("Name")
            if "Description" in columns:
                columns.remove("Description")

            # Insert at desired position
            columns.insert(insert_position, "Name")
            columns.insert(insert_position + 1, "Description")

            # Reorder DataFrame
            result_df = result_df[columns]

            self.logger.info("Successfully added gene descriptions")
            return result_df

        except Exception as e:
            # Sanitize error message for logging to prevent log injection
            safe_error = str(e).replace("\n", "").replace("\r", "")
            self.logger.error(f"Failed to add gene descriptions: {safe_error}")
            raise

    def add_descriptions_to_excel_sheets(
        self,
        excel_file: str,
        combinations: List[str],
        output_file: str,
        gene_column: str = "Gene",
    ) -> None:
        """
        Add gene descriptions to multiple Excel sheets.

        Args:
            excel_file: Path to input Excel file
            combinations: List of sheet names to process
            output_file: Path to output Excel file
            gene_column: Name of the column containing gene IDs
        """
        try:
            # Sanitize file path for logging to prevent log injection
            safe_excel_file = str(excel_file).replace("\n", "").replace("\r", "")
            self.logger.info(f"Processing Excel file: {safe_excel_file}")

            # Get gene descriptions
            self.gene_service.fetch_gene_descriptions()

            with pd.ExcelWriter(output_file) as writer:
                for combination in combinations:
                    try:
                        # Read sheet
                        df = pd.read_excel(excel_file, sheet_name=combination)

                        # Add descriptions
                        annotated_df = self.add_descriptions_to_dataframe(df, gene_column=gene_column)

                        # Write to new file
                        safe_combination = str(combination).replace("\n", "").replace("\r", "")
                        annotated_df.to_excel(writer, sheet_name=safe_combination, index=False)

                    except Exception as e:
                        # Sanitize error message for logging to prevent log injection
                        safe_error = str(e).replace("\n", "").replace("\r", "")
                        safe_combination = str(combination).replace("\n", "").replace("\r", "")
                        self.logger.warning(f"Failed to process sheet {safe_combination}: {safe_error}")
                        continue

            # Sanitize file path for logging to prevent log injection
            safe_output_file = str(output_file).replace("\n", "").replace("\r", "")
            self.logger.info(f"Successfully created annotated Excel file: {safe_output_file}")

        except Exception as e:
            # Sanitize error message for logging to prevent log injection
            safe_error = str(e).replace("\n", "").replace("\r", "")
            self.logger.error(f"Failed to process Excel file: {safe_error}")
            raise

    def add_descriptions_to_functional_annotation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add gene names to functional annotation results.

        This method handles the specific case where genes are stored as
        comma-separated values in a 'Genes' column.

        Args:
            df: DataFrame with functional annotation results

        Returns:
            DataFrame with added Gene_Name column
        """
        try:
            self.logger.info("Adding gene names to functional annotation")

            # Get gene mapping
            gene_mapping = self.gene_service.get_gene_mapping()

            # Create a copy of the DataFrame
            result_df = df.copy()

            gene_names = []
            for _, row in result_df.iterrows():
                if "Genes" in row and pd.notna(row["Genes"]):
                    genes = str(row["Genes"]).split(",")
                    names = []

                    for gene in genes:
                        gene = gene.strip()
                        if gene in gene_mapping:
                            name = gene_mapping[gene].get("name", gene)
                            if pd.notna(name) and name != "":
                                names.append(name.upper())
                            else:
                                names.append(gene)
                        else:
                            names.append(gene)

                    gene_names.append(",".join(names))
                else:
                    gene_names.append("")

            result_df["Gene_Name"] = gene_names

            self.logger.info("Successfully added gene names to functional annotation")
            return result_df

        except Exception as e:
            # Sanitize error message for logging to prevent log injection
            safe_error = str(e).replace("\n", "").replace("\r", "")
            self.logger.error(f"Failed to add gene names to functional annotation: {safe_error}")
            # Return original DataFrame if annotation fails
            return df
