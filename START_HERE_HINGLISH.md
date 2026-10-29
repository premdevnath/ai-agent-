# Financial RAG Chatbot — Start Here (Windows)

Ye project financial PDFs ko read karta hai, unke text ko searchable chunks mein convert karta hai, aur OpenAI + LangChain + ChromaDB se document-based answers deta hai. Answer ke saath source PDF aur page number dikhane ke liye design kiya gaya hai.

## Tumhe kya chahiye?

- Windows 10/11
- Python 3.11 ya 3.12
- Internet connection (pehli baar packages install karne aur OpenAI API use karne ke liye)
- OpenAI API key with API billing/credits enabled

Docker, database server, cloud hosting, Git, ya alag se Chroma install karne ki zaroorat nahi hai. ChromaDB local folder mein store hota hai.

## Sirf ye steps follow karo

1. ZIP ko **Extract All** karo. Folder ko kisi simple location par rakho, e.g. `C:\AIProjects\financial-rag-chatbot-2026-fixed`.
2. Agar Python installed nahi hai, official page se Python 3.12 install karo: https://www.python.org/downloads/windows/ . Installation ke waqt **Add python.exe to PATH** tick karo.
3. Project folder ke andar `RUN_WINDOWS.bat` par double-click karo.
4. Pehli baar launcher virtual environment banayega aur required packages install karega. Ismein kuch minutes lag sakte hain. Terminal band mat karna.
5. Browser mein app khulne par left sidebar ke **OpenAI API Key** box mein apni key paste karo.
6. **Upload** se ek text-based financial PDF select karo.
7. **Process Documents** click karo. Indexing complete hone par chat mein question type karo.
8. App band karne ke liye terminal mein `Ctrl + C` dabao.

## Example questions

- What was the year-over-year revenue growth?
- Summarize the key risk factors mentioned in this report.
- What did management say about AI investments?
- Compare the operating margins between the two segments.

## Important notes

- API key app ke password field mein enter hoti hai; project source code mein key likhne ki zaroorat nahi hai. Key ko GitHub, screenshots, ya kisi ko chat mein share mat karna.
- OpenAI API usage can cost money. ChatGPT subscription does not automatically include API credits. Check your API billing and usage limits.
- Only PDFs containing selectable text are supported. Scanned/image-only PDFs need OCR, which is not included.
- The app answers based on retrieved text and may still make mistakes. Verify important financial figures in the cited source pages.
- This is a local single-user portfolio project, not a security-audited enterprise service.
- The original open-source project was created by Satya Sai Prakash Kantamani. This package includes local-runner and error-handling improvements; keep the original attribution when publishing derivative work.

## If something fails

- **Python not found:** Install Python 3.11/3.12, tick “Add python.exe to PATH”, then reopen the project and rerun `RUN_WINDOWS.bat`.
- **Invalid API key / 401:** Check the key and whether API billing/credits are active.
- **Quota / 429:** Check OpenAI API usage, billing, and rate limits.
- **PDF has no text:** Try a different PDF with selectable text rather than a scanned image.
- **Package install error:** Check internet access. Keep the terminal open and share the final error text for diagnosis.
