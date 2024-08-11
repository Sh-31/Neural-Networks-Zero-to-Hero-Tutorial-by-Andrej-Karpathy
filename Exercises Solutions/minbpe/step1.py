class BasicTokenizer:
    def train(self, text, vocab_size, verbose=False):
        tokens = [ord(i) for i in text]
        num_merges = abs(vocab_size - 256)
        self.num_merges = num_merges

        assert (vocab_size < 256) == False , "Can't use vocab size less then 256" 

        def get_pairs(tokens):
            pairs = {}
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] = pairs.get((tokens[i], tokens[i + 1]) , 0 ) + 1
            max_occ = max(pairs.items(), key=lambda x: x[1])   
            return  pairs , max_occ

        self.get_pairs = get_pairs

        def merge(tokens, pair, idx):
             new_token_list = []
             i = 0
             while i < len(tokens) - 1:
                if (tokens[i],tokens[i+1]) == pair[0]:
                    new_token_list.append(idx)
                    i += 2  
                else:
                    new_token_list.append(tokens[i])
                    i += 1

             if i == len(tokens) - 1:  
                new_token_list.append(tokens[i])
             return new_token_list

        def debyte(idx, marges_rev):
            if idx > 256:
                p1 , p2 = marges_rev[idx]
                return debyte(p1, marges_rev) + debyte(p2, marges_rev)
            return bytes([idx])

        new_token_idx = 257
        new_tokens = list(tokens)
        marges = {}
        marges_rev = {}

        while num_merges:
           pairs , max_pair_oc = get_pairs(new_tokens)
           new_tokens = merge(new_tokens, max_pair_oc, new_token_idx)
           marges[max_pair_oc[0]] = new_token_idx
           marges_rev[new_token_idx] = max_pair_oc[0]
          
           if verbose:
            print(f"merging {max_pair_oc} into a new token {new_token_idx}")
            print(f"merging {debyte(idx=new_token_idx, marges_rev=marges_rev)} into a new token {new_token_idx}")

           new_token_idx += 1
           num_merges-=1

        self.marges = marges
        return marges
        
    def encode(self, text):
       tokens = list(text.encode("utf-8"))
       while len(tokens) >= 2:
        _ , stats = self.get_pairs(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
       return tokens

    def decode(self, ids) :
        return b"".join(map(debyte, ids)).decode("utf-8" , errors='replace')


if __name__ == "__main__":
   Tokenizer =  BasicTokenizer()

   with open("taylorswift.txt", "r", encoding="utf-8") as f:
    text = f.read()

   new_vocab_size = 300
   Tokenizer.train(text=text, vocab_size=new_vocab_size, verbose=True)



    

