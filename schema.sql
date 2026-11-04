CREATE DATABASE IF NOT EXISTS tumor_seg CHARACTER SET utf8mb4;
USE tumor_seg;

CREATE TABLE IF NOT EXISTS predictions (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  original_name VARCHAR(255) NOT NULL,
  image_file    VARCHAR(255) NOT NULL,
  mask_file     VARCHAR(255) NOT NULL,
  overlay_file  VARCHAR(255) NOT NULL,
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
