#include "euclideanMst/custom/sql.h"

using namespace std;

// Function to execute an SQL command
void execute_sql_statement(sqlite3* db, const std::string& sql) {
    char* errmsg;
    int rc = sqlite3_exec(db, sql.c_str(), 0, 0, &errmsg);

    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << errmsg << std::endl;
        sqlite3_free(errmsg);
    }
}

// Function to write data to the SQLite database
void write_to_database(const std::string& dbFilename, int num_points, double mst_length, double normalized_mst_length) {
    sqlite3* db;
    int rc = sqlite3_open(dbFilename.c_str(), &db);

    if (rc) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    // Set a busy timeout of 5 seconds (5000 milliseconds)
    sqlite3_busy_timeout(db, 5000);

    // Create a table if it doesn't exist
    std::string createTableSQL = "CREATE TABLE IF NOT EXISTS data ("
                                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                "num_points INTEGER, "
                                "mst_length REAL, "
                                "normalized_mst_length REAL);";
    execute_sql_statement(db, createTableSQL);

    // Begin transaction to avoid locking issues during multiple operations
    execute_sql_statement(db, "BEGIN TRANSACTION;");

    // Prepare SQL insert statement
    std::string insertSQL = "INSERT INTO data (num_points, mst_length, normalized_mst_length) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(db, insertSQL.c_str(), -1, &stmt, 0);

    if (rc != SQLITE_OK) {
        std::cerr << "Can't prepare SQL statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return;
    }

    // Bind values to the SQL statement
    sqlite3_bind_int(stmt, 1, num_points);
    sqlite3_bind_double(stmt, 2, mst_length);
    sqlite3_bind_double(stmt, 3, normalized_mst_length);

    // Execute the SQL statement with retry logic if the database is locked
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cerr << "Execution failed: " << sqlite3_errmsg(db) << std::endl;
    }

    // Finalize the statement to release resources
    sqlite3_finalize(stmt);

    // Commit the transaction
    execute_sql_statement(db, "COMMIT;");

    // Close the database connection
    sqlite3_close(db);
}
