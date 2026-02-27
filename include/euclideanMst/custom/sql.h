#ifndef EUCLIDEAN_MST_CUSTOM_SQL_H
#define EUCLIDEAN_MST_CUSTOM_SQL_H

#include <string>
#include <sqlite3.h>

// Function to execute an SQL command
void execute_sql_statement(sqlite3* db, const std::string& sql);

// Function to write data to the SQLite database
void write_to_database(const std::string& dbFilename, int num_points, double mst_length, double normalized_mst_length);

#endif // EUCLIDEAN_MST_CUSTOM_SQL_H