# db.py

import re
import mysql.connector
from mysql.connector import Error
from datetime import datetime


# --------------------------------------------------
# DB CONNECTION
# --------------------------------------------------
def get_connection():
    try:
        return mysql.connector.connect(
            host="database-1.c9qgwy2si9n0.eu-north-1.rds.amazonaws.com",      # eg: pdf-plumber-db.xxxx.rds.amazonaws.com
            user="pdfplumber_admin",
            password="********",
            database="database-1"
        )
    except Error as e:
        print("DB connection error:", e)
        return None


# --------------------------------------------------
# SAFE CONVERSIONS
# --------------------------------------------------
def safe_int(val, default=1):
    try:
        return int(val)
    except:
        return default


def safe_float(val, default=0.0):
    try:
        return float(re.sub(r"[^\d.]", "", str(val)))
    except:
        return default


def safe_date(val):
    try:
        return datetime.strptime(val, "%d/%m/%Y").date()
    except:
        return None


# --------------------------------------------------
# INSERT PDF + BILL + ITEMS
# --------------------------------------------------
def insert_pdf_invoice(info, file_name, s3_path):
    conn = get_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()

        # 1️⃣ Insert PDF
        cursor.execute(
            "INSERT INTO pdf (pdf_name, s3_path) VALUES (%s, %s)",
            (file_name, s3_path)
        )
        pdf_id = cursor.lastrowid

        # 2️⃣ Insert Bill
        cursor.execute("""
            INSERT INTO bill (
                pdf_id, invoice_number, invoice_date,
                total_amount, vendor_name, customer_name, raw_text
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            pdf_id,
            info.get("invoice_number"),
            safe_date(info.get("invoice_date")),
            safe_float(info.get("total_amount")),
            info.get("vendor_name"),
            info.get("full_customer"),
            info.get("raw_text", "")
        ))
        bill_id = cursor.lastrowid

        # 3️⃣ Insert Bill Items
        for item in info.get("line_items", []):
            cursor.execute("""
                INSERT INTO bill_item (
                    bill_id, description, quantity, amount
                )
                VALUES (%s, %s, %s, %s)
            """, (
                bill_id,
                item.get("description"),
                safe_int(item.get("quantity")),
                safe_float(item.get("amount"))
            ))

        conn.commit()
        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print("Insert error:", e)
        conn.rollback()
        return False


# --------------------------------------------------
# FETCH DATA
# --------------------------------------------------
def fetch_all_pdfs():
    conn = get_connection()
    if conn is None:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM pdf ORDER BY pdf_id DESC")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print("Fetch pdf error:", e)
        return []
