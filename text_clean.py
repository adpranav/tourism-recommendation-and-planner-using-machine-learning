final_compile=[]
for i in range(42):
    review1=[]                      
    df_new=df[df["Place"]==column_values[i]]
    for i in df_new["Review"]:
        review1.append(i)
    
    
    
    def sentence_tokenize(arr):               ## passing array of paragraphs ( paragraphs in turn contain many sentences)
        final=[]
        for i in range(len(arr)):
            a=sent_tokenize(arr[i])
            for j in a:
                final.append(j)
        return final
    
    sentences1=sentence_tokenize(review1)
    
    
    
    def words_tokenizer(arr):          ## pass arrays of senteces
        word_list=[]
        for i in range(len(arr)):
            words=word_tokenize(arr[i].lower())
            for i in words:
                word_list.append(i)
        return word_list

    words1=words_tokenizer(sentences1)
    
    cleaned_words1=[w for w in words1 if not w in stop]
    
    
    
    import re

    def remove_unwanted_characters(words):
        cleaned_words = []
        pattern = r"[^\w\s]"  # Regex pattern to match non-alphanumeric characters

        for word in words:
            cleaned_word = re.sub(pattern, "", word)
            if cleaned_word:  # Check if the word is not empty after removing characters
                cleaned_words.append(cleaned_word)

        return cleaned_words
    
    
    final_cleaned1=remove_unwanted_characters(cleaned_words1)
    
    def join_words(word_array):
        word_list = ','.join(word_array).split(',')
        return ' '.join(word_list)
    
    sentence1=join_words(final_cleaned1)
    
    
    
    final_compile.append(sentence1)
    