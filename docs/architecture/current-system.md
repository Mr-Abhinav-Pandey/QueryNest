Querynest solves the problem-
Finding similar context words based on their meaning rather than exact spelling. If you use find and search tool, then you know that sometimes synonyms of words or closely related words cannot be found with it. For eg: you search tire and the uploaded documents contain car,road,etc. 
Then using semantic search technique using rag and vectordatabase we can find those related items too.

FLow -
User -> Upload PDF (can add more than one) -> Database updated -> Vector embeddings made -> Search -> functions and backend show result

Folder purpose-

backend - to write functions and environemnt and connecting supabase with python. Searching top - k retrieval fucntion.
frontend - 
    dart_tool ?
    android,ios,linux,macos,windows - flutter and app running commands platform wise
    lib - app code written for dart
    test - widget running test of website
    web - no idea
    runner - installing dependencies and running code in other devices

API inventory -
Post - to upload pdf, get - to search , and one more

Data flow -
frontend -> flutter -> supabase -> faiss -> llm -> python -> frontend 