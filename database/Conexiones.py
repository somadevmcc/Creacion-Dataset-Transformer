import psycopg2
from psycopg2 import extras
import os
import dotenv
from dotenv import load_dotenv
from pathlib import Path


dotenv.load_dotenv('/code/.env', encoding='utf-8')



# Database connection parameters
db_params = {
    "host": os.getenv('DBIP'),
    "port": os.getenv('DBPORT'),
    "dbname": os.getenv('DBNAME'),
    "user": os.getenv('DBUSER'),
    "password": os.getenv('DBPASS')
    
}


def conexion():
    # Establish a connection to the PostgreSQL database
    try:
        connection = psycopg2.connect(**db_params)

        # Create a cursor object to interact with the database
        cursor = connection.cursor()
        cur = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        return cur,connection

        
       

    except psycopg2.Error as error:
        print("Error connecting to the database:", error)
def commit(connection):
    if connection:
        try:
            connection.commit()
        except psycopg2.Error as error:
            print("Error committing changes to the database:", error)