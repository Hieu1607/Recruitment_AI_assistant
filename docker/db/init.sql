-- Simple initialization script
CREATE TABLE IF NOT EXISTS initial_setup (
    id SERIAL PRIMARY KEY,
    setup_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
