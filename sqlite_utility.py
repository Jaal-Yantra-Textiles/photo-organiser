import sqlite3
from utility import serialize_array
from logger_util import logger


class SQLiteDB:
    def __init__(self, db_path):
        """
        Initialize the SQLite database path.
        """
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)


    @property
    def cursor(self):
        return self.connection.cursor()

    def _connect(self):
        """
        Create and return a new SQLite connection.
        """
        return sqlite3.connect(self.db_path)
    

    def execute_query(self, query):
        """
        Execute a raw SQL query.
        """
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            conn.commit()
        except sqlite3.Error as error:
            print(f"Failed to execute query: {error}")

    def create_tables(self):
        """
        Create the necessary tables if they don't exist.
        """
        conn = self._connect()
        cursor = conn.cursor()

        # Create table for image features
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_features (
            id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL,
            face_embedding BLOB,
            color_feature BLOB,
            texture_feature BLOB,
            pattern_feature BLOB,
            image_roi BLOB
        )
        ''')

        # Create table for comparison scores
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS comparison_scores (
            image1_id INTEGER,
            image2_id INTEGER,
            face_distance REAL,
            color_distance REAL,
            texture_distance REAL,
            pattern_distance REAL,
            FOREIGN KEY (image1_id) REFERENCES image_features (id),
            FOREIGN KEY (image2_id) REFERENCES image_features (id)
        )
        ''')
        
        conn.commit()
        conn.close()

    def insert_image_features(self, image_path, face_embedding, color_feature, texture_feature, pattern_feature):
        """
        Insert image features into the database.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO image_features (image_path, face_embedding, color_feature, texture_feature, pattern_feature)
            VALUES (?, ?, ?, ?, ?)
        ''', (image_path, serialize_array(face_embedding) if face_embedding is not None else None, serialize_array(color_feature), serialize_array(texture_feature), serialize_array(pattern_feature)))

        conn.commit()
        conn.close()

    def insert_image_roi(self, image_path, serialized_roi):
        """
        Insert the ROI of an image into the database.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR IGNORE INTO image_features (image_path, image_roi)
            VALUES (?, ?)
        ''', (image_path, serialized_roi))
        conn.commit()

    def fetch_all_image_rois(self):
        """
        Fetch ROIs for all images.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT image_path, image_roi FROM image_features WHERE image_roi IS NOT NULL")
        results = cursor.fetchall()

        conn.close()
        return results
    
    def update_image_features(self, image_path, face_embedding, color_feature, texture_feature, pattern_feature):
        """
        Update features for an existing image in the database.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE image_features 
            SET face_embedding = ?, color_feature = ?, texture_feature = ?, pattern_feature = ?
            WHERE image_path = ?
        ''', (serialize_array(face_embedding) if face_embedding is not None else None, serialize_array(color_feature), serialize_array(texture_feature), serialize_array(pattern_feature), image_path))
        conn.commit()
        conn.close()

    def fetch_all_image_features(self):
        """
        Fetch features for all images.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM image_features")
        results = cursor.fetchall()

        conn.close()
        return results

    def insert_comparison_scores(self, image1_id, image2_id, face_distance, color_distance, texture_distance, pattern_distance):
        """
        Insert comparison scores between two images.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO comparison_scores (image1_id, image2_id, face_distance, color_distance, texture_distance, pattern_distance)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image1_id, image2_id, face_distance, color_distance, texture_distance, pattern_distance))

        conn.commit()
        conn.close()

    def fetch_all_comparison_scores(self):
        """
        Fetch comparison scores for all image pairs.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM comparison_scores")
        results = cursor.fetchall()

        conn.close()
        return results


    def close(self):
        """
        Close the database connection. (Not required in this refactored version but kept for compatibility)
        """
        pass


    def drop_table(self, table_name):
        """
        Drop a table from the database.
        """
        conn = self._connect()
        cursor = conn.cursor()

        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
            logger.info(f"Table {table_name} dropped successfully.")
        except Exception as e:
            logger.info(f"Error occurred while dropping table {table_name}: {str(e)}")

    def initialize_database(self):
        """
        Initialize (or re-initialize) the database by dropping existing tables and creating new ones.
        """
        try:
            # Drop tables
            self.drop_table("image_features")
            self.drop_table("comparison_scores")
            
            # Create tables
            self.create_tables()
            logger.info("Database initialized successfully.")
        except Exception as e:
            logger.info(f"Error occurred while initializing database: {str(e)}")

        
    def fetch_all_image_features_with_roi(self):
        """
        Fetch features and ROIs for all images.
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT id, image_path, image_roi, face_embedding, color_feature, texture_feature, pattern_feature FROM image_features")
        results  = cursor.fetchall()

        conn.close()

        return results
    
    
    def insert_cluster_label(self,image_id, cluster_label, pca_feature_1, pca_feature_2):
        """
        Insert or update the cluster label and PCA features for a given image ID.
        """
        conn = self._connect()
        cursor = conn.cursor()

        # Check if an entry for the given image_id already exists
        cursor.execute("SELECT * FROM ImageClusters WHERE image_id=?", (image_id,))
        entry = cursor.fetchone()

        if entry:
            # Update if the image ID is already present
            logger.info(type(cluster_label))
            cursor.execute(
                """
                UPDATE ImageClusters 
                SET cluster_label=?, pca_feature_1=?, pca_feature_2=?
                WHERE image_id=?
                """, 
                (cluster_label, pca_feature_1, pca_feature_2, image_id)
            )
        else:
            # Insert a new record if the image ID is not present
            cursor.execute(
                """
                INSERT INTO ImageClusters (image_id, cluster_label, pca_feature_1, pca_feature_2)
                VALUES (?, ?, ?, ?)
                """, 
                (image_id, cluster_label, pca_feature_1, pca_feature_2)
            )

        conn.commit()
        cursor.close()
        conn.close()



    def fetch_clustered_data_with_features(self):
        """
        Fetch cluster labels and PCA reduced features.
        """
        conn = self._connect()
        cursor = conn.cursor()

        query = """
        SELECT image_id, cluster_label, pca_feature_1, pca_feature_2 
        FROM ImageClusters
        """
        cursor.execute(query)
        return cursor.fetchall()