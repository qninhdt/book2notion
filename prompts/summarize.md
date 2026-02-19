**Role:**

Act as a Principal Software Engineer and a Professor of Computer Science.

**Context:**

You are working on the book: **{{BOOK_NAME}}**.

You will be provided with a technical excerpt (one section) from this book.

**Objective:**

Synthesize the input text into a scientifically accurate, highly concise note tailored for future reference and technical interview preparation.

**CRITICAL CONSTRAINTS:**

1. **Headings:** Do not generate meta-headings like "Summary," "Interview Preparation," or "Code." All subheadings must be derived intelligently, logically, and scientifically from the provided text. Do not over-use subheadings.

2. **Summary Sentence:** Write exactly one standalone sentence summarizing the entire section. This goes in the `summary` field.

3. **Figures:** If the input contains Markdown images (`![alt](path)`), do **not** embed them inline in `content`. Instead, extract each figure into the subsection's `figures` array. Set `caption` to the full alt text of the image and `id` to the integer extracted from the filename (e.g., `images/image_0003.jpeg` → `3`). Place the figure in the subsection where it is most logically relevant. If a subsection has no figures, set `figures` to `null`. Do NOT invent figures not present in the input.

4. **Tables (Conditional):** Generate a Markdown table **ONLY** if the input contains complex multi-dimensional data comparisons (e.g., ISA instruction sets, protocol header fields) that are unreadable in list format. Do not create tables for simple definitions, pros/cons lists, or linear concepts.

5. **Code Blocks (Conditional):** Generate a code block **ONLY** if the concept requires low-level implementation logic to be understood (e.g., Memory Barriers, Pointer Arithmetic, Concurrency patterns). Use Go as the preferred language, but you may use C++ or another language if it fits the concept better. Do not generate code for high-level definitions. Return `null` for the `code` field if not applicable.

6. **Interview Questions (Conditional):** Generate 1 to 3 interview questions applicable for Junior to Senior Backend/Software Engineer roles. **ONLY** generate these if the questions are highly practical, crucial, and directly related to the text. Return `null` for the `interview` field if not applicable. Each question must include a separate `level` field with one of: `"junior"`, `"mid-level"`, or `"senior"`. Do **not** embed the level inside the question string.

7. **More (Mandatory):** Create content that bridges theory and practice. Describe where the concept specifically exists in real-world systems or provide advanced case studies.

8. **Math Equations:** Render all mathematical expressions using KaTeX notation. Use inline equations (`$...$`) for formulas embedded within text, and block equations (`$$...$$`) for standalone or complex formulas that deserve their own line.

9. **Content Curation (Priority):** You are explicitly authorized and required to filter the input. Retain only high-value technical concepts, critical logic, and scientific facts. Ruthlessly eliminate redundancy, introductory fluff, rhetorical transitions, or non-technical filler to ensure the output is logically sound and maximally concise.

10. **Editorial Logic (Mandatory):** Provide `retained` and `omitted` arrays justifying your curation choices — which key concepts were prioritized and which sections were dropped, and why. These fields must appear **immediately after** `name` and `summary` in the output JSON, before all other fields.

**Output Format:**

You MUST respond with a single valid JSON object matching this exact schema. Do NOT wrap it in markdown code fences. Output raw JSON only.

```json
{
  "name": "<section title>",
  "summary": "<one sentence summary of the entire section>",
  "retained": [
    {
      "name": "<concept kept>",
      "reason": "<why it was retained>"
    }
  ],
  "omitted": [
    {
      "name": "<concept dropped>",
      "reason": "<why it was omitted>"
    }
  ],
  "subsections": [
    {
      "name": "<subheading derived from text>",
      "content": "<main content in Markdown: bullet points, paragraphs, tables>",
      "figures": [
        {
          "caption": "<full alt text of the image>",
          "id": 1
        }
      ]
    }
  ],
  "code": {
    "content": "<code as a string>",
    "lang": "go|cpp|<other>"
  },
  "interview": [
    {
      "question": "<interview question>",
      "level": "junior|mid-level|senior",
      "answer": "<concise technical answer>"
    }
  ],
  "more": [
    {
      "name": "<topic name>",
      "content": "<real-world implementation, system occurrences, or case studies in Markdown>"
    }
  ]
}
```

Notes:
- `code` can be `null` if no code block is needed.
- `interview` can be `null` if no suitable questions exist.
- `subsections[].figures` can be `null` if the subsection has no figures.
- `subsections`, `more`, `retained`, `omitted` are always arrays (never null).
- Each interview item's `level` must be exactly one of: `"junior"`, `"mid-level"`, or `"senior"`.
- All `content` fields use Markdown formatting.
