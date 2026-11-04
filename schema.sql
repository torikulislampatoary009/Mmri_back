-- Reference only: the app creates these tables itself on first connection.
-- (Aiven gives you a ready-made database called defaultdb, so no CREATE DATABASE is needed.)
CREATE TABLE IF NOT EXISTS predictions (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  original_name VARCHAR(255) NOT NULL,
  width         INT NOT NULL,
  height        INT NOT NULL,
  tumor_pixels  INT NOT NULL,
  tumor_percent FLOAT NOT NULL,
  threshold_val FLOAT NOT NULL,
  demo_mode     TINYINT(1) NOT NULL DEFAULT 0,
  inference_ms  INT NOT NULL,
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_created (created_at)
);
CREATE TABLE IF NOT EXISTS prediction_files (
  prediction_id INT NOT NULL,
  kind          VARCHAR(10) NOT NULL,
  data          MEDIUMBLOB NOT NULL,
  PRIMARY KEY (prediction_id, kind),
  FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);
