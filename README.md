# Table of Contents
- Overview
- Features
- Approaches
   - 1. Sentence Transformer Semantic Chunking
   - 2. Recursive Character Text Splitter
   - 3. Custom Prompt Generation
   - 4. Graph Database (Neo4j) + NER
   - 5. Hybrid Approach: Sentence Transformer + Recursive Splitter
- Evaluation
- Best Performing Solution
- Future Work


## Research Report: Optimizing RAG Retrieval and Graph Database Integration for Enhanced Data Ingestion

### Introduction

As a **researcher and developer**, my objective was to improve the **Retrieval-Augmented Generation (RAG)** system for a client, focusing on enhancing their **data ingestion pipeline** by combining advanced **chunking strategies** with **graph databases** to map relationships between entities. The final goal was to develop an **efficient retrieval system** that not only improves accuracy but also supports **real-time applications** where possible.

During the course of this research, I explored multiple methods such as **sentence transformer-based chunking**, **recursive splitting**, **custom prompt generation**, and integrating **Neo4j (graph database)** to handle relational data using **Named Entity Recognition (NER)**. While each approach had its strengths, combining **RAG with a graph database** produced the best retrieval accuracy, especially when mapping relationships between entities was crucial. However, this method proved **time-consuming** and less feasible for real-time systems.

### Objective

The key objectives were:
1. **Improve semantic accuracy** and **preserve context** in document retrieval.
2. **Optimize retrieval time** for scalability.
3. **Integrate NER and graph databases** to map complex relationships in data.
4. **Evaluate feasibility** for both real-time applications and larger data contexts.

### Approach and Methodology

#### Initial Benchmark: Basic RAG with Recursive Character Splitter
To start, I implemented a basic RAG retrieval system using LangChain's **recursive character text splitter**. This method, while efficient in speed, lacked the **semantic coherence** needed for accurate retrieval, often splitting context mid-sentence or between related topics.

#### Advanced Approaches Tested

1. **Sentence Transformer Semantic Chunking**:
   - **Tools**: Hugging Face Sentence Transformers
   - **Method**: This technique involved creating **semantic embeddings** for sentences and clustering similar sentences together. This ensured that context was maintained across document chunks, leading to better retrieval accuracy.
   - **Performance**: It performed well in preserving **semantic meaning**, but the process was **computationally expensive** and slow, making it impractical for real-time applications.

   ![Semantic Chunking](https://via.placeholder.com/600x300?text=Semantic+Chunking+Process)

2. **Recursive Character Text Splitter (LangChain)**:
   - **Tools**: LangChain’s Recursive Character Splitter
   - **Method**: The text was split at character boundaries to keep chunks within a specific size, regardless of semantic meaning. This approach offered **speed**, but the lack of contextual awareness caused issues with fragmented retrieval.
   - **Performance**: While **fast**, the retrieval accuracy was compromised.

   ![Recursive Character Splitter](https://via.placeholder.com/600x300?text=Recursive+Text+Splitting)

3. **Custom Prompt Generation**:
   - **Method**: Generated custom prompts based on user queries, chunking documents according to the generated prompts.
   - **Performance**: This method was limited by the **context window** size of the LLMs. Though it allowed for better understanding, the context window became a bottleneck, reducing retrieval efficiency for larger documents.

   ![Custom Prompt Generation](https://via.placeholder.com/600x300?text=Custom+Prompt+Generation)

4. **Graph Database Integration (Neo4j) + NER**:
   - **Tools**: Neo4j, Hugging Face NER models
   - **Method**: For data that contained **relationships between entities**, I integrated a **graph database (Neo4j)**. **Named Entity Recognition (NER)** was used to identify key entities in the text, and the relationships between them were mapped into a graph structure. This allowed for much more accurate retrieval when querying complex, interconnected data.
   - **Performance**: **Highest retrieval accuracy** was achieved with this method, especially for documents with multiple related entities. However, the time cost was significant, as constructing the graph and mapping relationships was **time-intensive**.
   
   - **Best Use Case**: This method is ideal when there is a need to **preserve relationships** between entities in the data, such as in legal, medical, or research documents.

   ![Graph Database and NER Integration](https://via.placeholder.com/600x300?text=Graph+DB+Integration+with+NER)

5. **Hybrid Approach: Sentence Transformer + Recursive Splitter**:
   - **Method**: Combined the **semantic chunking** of the sentence transformer with the **speed** of recursive splitting. Text was initially split into manageable chunks and then semantically clustered.
   - **Performance**: This method provided a **balanced** solution with **faster retrieval** times and **good semantic accuracy**. It fit well within the **LLM context window**, making it an effective solution for most real-time applications.

   ![Hybrid Approach](https://via.placeholder.com/600x300?text=Hybrid+Approach+Using+Semantic+and+Recursive+Chunking)

### Evaluation Metrics

1. **Cosine Similarity**: I used cosine similarity between sentence embeddings to objectively measure how well the retrieved chunks matched the user's query.
2. **Human Evaluation**: Evaluators assessed the quality of retrieval, particularly the **contextual accuracy** of the chunks returned by each method.

### Best Performing Solution: Graph Database + RAG

The **Graph Database (Neo4j) combined with RAG** was the **best-performing solution** in terms of retrieval accuracy, particularly when dealing with complex datasets where relationships between entities were important. **NER models** were employed to extract entities and map relationships, which significantly improved retrieval by **capturing the interconnectedness** of the data.

However, this approach was the **most time-consuming**, as constructing and querying a graph database requires significant computational resources, making it less suitable for **real-time applications**. Despite the high time cost, it remains the best choice for **datasets with complex entity relationships**, where accuracy and context preservation are paramount.

### Integration with Azure Document Intelligence

For the client's project, **Azure Document Intelligence** was integrated to intelligently extract the most relevant portions of the document before applying the retrieval strategies. By combining **Azure's document extraction** capabilities with **advanced chunking** and **graph-based retrieval**, the system was optimized for **better performance** and **contextual accuracy**.

### Key Findings

- **Graph Database (Neo4j) + NER** provided the highest retrieval accuracy but was time-consuming, making it suitable for complex datasets with entity relationships, such as **legal** or **medical records**.
- **Sentence Transformer Semantic Chunking** was accurate but too slow for real-time applications.
- **Recursive Character Text Splitter** offered speed but lacked context preservation.
- The **Hybrid Approach** (sentence transformer + recursive splitter) balanced **efficiency** and **semantic accuracy**, making it the most practical solution for general use cases.
- **Custom Prompt Generation** was limited by LLM context windows, making it less feasible for large datasets.

### Conclusion

This research demonstrates the potential of combining **advanced chunking methods** with **graph databases** for improving RAG retrieval, particularly in scenarios involving complex relationships between entities. While the **Graph Database + NER** approach is ideal for accuracy, the **Hybrid Approach** provides a more **scalable** solution for real-time applications with constrained LLM context windows.

The final system, integrating **Azure Document Intelligence** and **graph database retrieval**, significantly improved the client’s **data ingestion pipeline**, ensuring more accurate and efficient document retrieval.

### Future Work

1. **Optimizing Graph Database Retrieval**: Investigating more efficient ways to integrate graph databases with RAG to reduce time costs.
2. **Distributed Processing**: Leveraging distributed computing to speed up the semantic chunking process.
3. **Adaptive Chunking Strategies**: Dynamically adjusting the chunking approach based on query complexity and document structure.



