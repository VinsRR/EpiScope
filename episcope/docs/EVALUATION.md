# Evaluation Protocol

Measuring the quality of answers and extractions produced by EpiScope is
critical for ensuring reliability.  We outline a protocol to assess
both the **Explorer** and **PrecisionMiner** components.

## Explorer Evaluation

1. **Question Set Preparation** – Curate a list of diverse questions
   covering different pathogens, data modalities and complexity levels.
   Each question should have a known answer derivable from the
   available papers.

2. **Ground Truth** – For each question identify the correct answer
   and supporting passages in the source papers.  Annotate the paper
   identifiers, section types and snippet boundaries.

3. **Automated Runs** – Use the CLI or API to execute the queries
   against the Explorer pipeline.  Collect the generated answer and
   provenance for each question.

4. **Metrics**:

   * **Answer Accuracy** – Compare the generated answer to the ground
     truth at the sentence level.  Compute precision/recall/F1 for
     factoids or use ROUGE/BLEU metrics for free‑text answers.
   * **Citation Precision** – Compute the fraction of returned
     evidences that actually support the answer.  A high citation
     precision indicates that the retrieval step is working well.
   * **Citation Recall** – Compute the fraction of ground‑truth
     evidences that were returned.  This measures the coverage of
     retrieval.
   * **Latency** – Measure average response time for retrieval and
     generation.

5. **Human Judgement** – For a subset of questions, ask domain
   experts to grade the answers for correctness and completeness.  Use
   a Likert scale to capture confidence.

## PrecisionMiner Evaluation

1. **Dataset** – Prepare a corpus of PDFs with known ground truth
   extractions (paper type classifications, tables, references,
   structured sections).

2. **Parsing Accuracy** – Use the GROBID output to compare the
   extracted sections and references against hand‑annotated ground
   truth.  Compute exact match and partial match scores.

3. **Extraction Modules**:

   * **Paper Classifier** – Evaluate accuracy and F1 of the
     classification into literature review vs. data analysis.  Use
     cross‑validation on a labelled dataset.
   * **LLM Extractor** – For each extracted data source, compare the
     extracted DOI/URL/accession numbers against ground truth.  Compute
     precision and recall.
   * **Table Extractor** – Compare the number and content of extracted
     tables with manual extraction.
   * **Reference Extractor** – Measure the recall of references and
     accuracy of identifying data sources among them.

4. **End‑to‑End** – Run the full PrecisionMiner pipeline on a set of
   papers and compare the final CSV/JSON outputs to manually curated
   results.  Compute entity‑level F1 scores.

5. **Performance** – Measure average processing time per paper and
   resource utilisation (CPU, memory) during extraction and indexing.
