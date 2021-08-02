float balance_heuristic(float pdf, float pdf_other) {
  return pdf / (pdf + pdf_other);
}
