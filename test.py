# test_db.py
import mysql.connector

conn = mysql.connector.connect(
    host="database-1.c9qgwy2si9n0.eu-north-1.rds.amazonaws.com",
    user="pdfplumber_admin",
    password="*******",
    database="database-1"
)

print("CONNECTED SUCCESSFULLY")
conn.close()
