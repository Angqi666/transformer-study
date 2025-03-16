import jsonlines  
import csv        
import itertools  


def pretrain_process(chunk_size=50000, max_entries=100000):
    """
    Preprocessor function: read data from JSON Lines file, write eligible text to CSV file.
    
    Parameters.
    - chunk_size: the number of records to be processed at a time, default value is 50000
    
    Procedure: 1.
    1. Open the JSON Lines file and the target CSV file. 2.
    2. Read the data in chunks.
    3. For each record:
       - Get the content of the 'text' field;
       - If the length of the text is greater than 512, the record is skipped;
       - If the text is longer than 512, skip the record; otherwise, write the text to the CSV file. 4.
    4. catch and handle possible UnicodeDecodeError error and output the error message. 5.
    5. Output the progress of each block after processing.
    """
    
    chunk_idx = 0
    entries_processed = 0

    with jsonlines.open('/egr/research-slim/shared/LLM_data/mobvoi_seq_monkey_general_open_corpus.jsonl') as reader:
        with open('/egr/research-slim/liangqi1/LLM/transformer-study/data/pretrain_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['text'])

            while entries_processed < max_entries:
                chunk = list(itertools.islice(reader, chunk_size))
                if not chunk:
                    break

                for idx, obj in enumerate(chunk):
                    if entries_processed >= max_entries:
                        break
                    try:
                        content = obj.get('text', '')
                        if len(content) > 512:
                            continue
                        writer.writerow([content])
                        entries_processed += 1
                    except UnicodeDecodeError as e:
                        print(f"Skipping invalid line {chunk_idx * chunk_size + idx + 1}: {e}")
                        continue
                chunk_idx += 1
                print('chunk:', ((chunk_idx - 1) * chunk_size, chunk_idx * chunk_size), 'process end')

# 运行预处理函数，最多处理100000条记录
# Run the preprocessor function to process up to 100,000 records
if __name__ == "__main__":
    pretrain_process(chunk_size=50000, max_entries=100000)
