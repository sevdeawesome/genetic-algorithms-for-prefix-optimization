from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models import load_model, get_target_prob
import math

import numpy as np
import torch

torch.set_printoptions(sci_mode=False)

class GA:
    def __init__(self, population_size:int, mutation_rate:float, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, prompt: str, target_token: str, prefix_length: int = 5, fitness_sharing: bool = False, sharing_sigma: float = 0.5, sharing_alpha: float = 1.0, crowding: bool = False, crowding_factor: int = 3):
        self.__population_size = population_size
        self.__mutation_rate = mutation_rate
        self.__tokenizer = tokenizer
        self.__model = model

        self.__prefix_length = prefix_length
        self.__prompt = prompt
        self.__target_token = target_token
        self.__target_token_ids = self.__tokenizer.encode(self.__target_token, add_special_tokens=False)[0]

        self.__device = next(model.parameters()).device
        self.__prompt_token_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False), device=self.__device)


        # niching stuff!
        self.__fitness_sharing = fitness_sharing
        self.__sharing_sigma = sharing_sigma  # niche radius thing
        self.__sharing_alpha = sharing_alpha  # shape param
        self.__crowding = crowding
        self.__crowding_factor = crowding_factor

        # get embedding layer (different for diff models)
        self.__embedding_layer = self._get_embedding_layer()

        self.__population = self.__initialize_population()

        print('model precision:', self.__embedding_layer.weight.dtype)

    def _get_embedding_layer(self):
        """try to find embedding layer for whatever model we have"""
        # GPT-2
        if hasattr(self.__model, 'transformer') and hasattr(self.__model.transformer, 'wte'):
            return self.__model.transformer.wte
        # LLaMA/Mistral
        elif hasattr(self.__model, 'model') and hasattr(self.__model.model, 'embed_tokens'):
            return self.__model.model.embed_tokens
        else:
            raise ValueError(f"Cannot find embedding layer for model type: {type(self.__model)}")

    def __initialize_population(self):
        # random tokens to start out with
        return torch.randint(0, self.__tokenizer.vocab_size, (self.__population_size, self.__prefix_length), device=self.__device)

    def __evaluate_fitness_batch(self, batch_inds):
        full_tokens = torch.cat([batch_inds, self.__prompt_token_ids.unsqueeze(0).repeat(batch_inds.size(0), 1)], dim=1)

        logits = self.__model(full_tokens).logits
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        target_probs = probs[:, self.__target_token_ids]

        return target_probs

    def __evaluate_fitness(self, population, batch_size = 64):
        fitness_scores = []

        for i in range(0, population.shape[0], batch_size):
            batch_inds = population[i : i + batch_size]
            batch_scores = self.__evaluate_fitness_batch(batch_inds)
            fitness_scores.append(batch_scores)

        raw_fitness = torch.cat(fitness_scores, dim=0)

        if self.__fitness_sharing:
            return self.__apply_fitness_sharing(raw_fitness, population)
        return raw_fitness

    def __compute_pairwise_distances(self, pop_vectors):
        """pairwise euclidean distances in embedding space"""
        N, L, D = pop_vectors.shape
        flat_vecs = pop_vectors.reshape(N, L * D)

        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        sq_norms = (flat_vecs ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        sq_dists = sq_norms + sq_norms.T - 2.0 * flat_vecs @ flat_vecs.T  # [N, N]
        sq_dists = torch.clamp(sq_dists, min=0.0)  # numerical stuff
        distances = torch.sqrt(sq_dists)

        # normalize
        max_dist = distances.max()
        if max_dist > 0:
            distances = distances / max_dist

        return distances

    def __apply_fitness_sharing(self, fitness, population):
        """fitness sharing for diversity"""
        N, L = population.shape

        pop_vectors = self.__ids_to_vectors(population)  # [N, L, D]

        distances = self.__compute_pairwise_distances(pop_vectors)  # [N, N]

        # sharing function
        sigma = self.__sharing_sigma
        alpha = self.__sharing_alpha

        sharing = torch.where(
            distances < sigma,
            1.0 - (distances / sigma) ** alpha,
            torch.zeros_like(distances)
        )

        niche_counts = sharing.sum(dim=1)  # [N]

        niche_counts = torch.clamp(niche_counts, min=1.0)

        shared_fitness = fitness / niche_counts

        return shared_fitness
            


    def __roulette_selection(self, fitnessScores, population, num_parents):
        # roulette wheel selection
        
        if num_parents % 2 != 0:
            num_parents += 1

        if num_parents > len(fitnessScores):
            raise ValueError("num_parents must be less than or equal to population size")
        
        if num_parents == len(fitnessScores):
            return population
        
        normalized_fitness = fitnessScores / fitnessScores.sum()
        selected_inds = torch.multinomial(normalized_fitness, num_parents, replacement=False)

        return population[selected_inds]

    def __ids_to_vectors(self, ids):
        return self.__embedding_layer(ids)
    
    def __vectors_to_ids_batch(self, vectors):
        # nearest neighbor to find closest tokens

        # vectors: [p, l, d]
        B, L, D = vectors.shape
        V, D2 = self.__embedding_layer.weight.shape
        assert D == D2

        vecs = vectors.reshape(B * L, D)                      # [N, d]
        emb = self.__embedding_layer.weight                   # [V, d]

        # [N], [V]
        vec_norm = (vecs ** 2).sum(dim=1, keepdim=True)      # [N, 1]
        emb_norm = (emb ** 2).sum(dim=1, keepdim=True).T     # [1, V]

        # [N, V] = N*1 + 1*V - 2 * [N,d] @ [d,V]
        dists = vec_norm + emb_norm - 2.0 * vecs @ emb.T     # [N, V]

        dist_idxs = dists.argmin(dim=1)                      # [N]
        return dist_idxs.view(B, L)                          # [p, l]

    def __vectors_to_ids(self, vectors, batch_size = 6):
        ids = []

        for i in range(0, vectors.shape[0], batch_size):
            batch_vectors = vectors[i : i + batch_size]
            batch_ids = self.__vectors_to_ids_batch(batch_vectors)
            ids.append(batch_ids)

        return torch.cat(ids, dim=0)

    def __crossover(self, selected_parent_vectors):
        # crossover in embedding space

        num_of_pairs = selected_parent_vectors.shape[0] // 2

        parent0 = selected_parent_vectors[0 : num_of_pairs]
        parent1 = selected_parent_vectors[num_of_pairs : num_of_pairs * 2]

        crossover_points0 = torch.rand((parent0.shape[0], parent0.shape[1], 1), device=self.__device)
        crossover_points1 = torch.rand((parent0.shape[0], parent0.shape[1], 1), device=self.__device)

        children0 = parent0 * crossover_points0 + parent1 * (1 - crossover_points0)
        children1 = parent0 * crossover_points1 + parent1 * (1 - crossover_points1)

        childrens = torch.cat([children0, children1], dim=0)

        return childrens

    def __culling(self, population, fitness_scores, population_size, elite_fraction=0.2):
        # keep best ones + some random for diversity

        num_elites = max(1, int(population_size * elite_fraction))
        num_random = population_size - num_elites

        topk_scores, topk_indices = fitness_scores.topk(num_elites, dim=0)
        elites = population[topk_indices]

        # rest is fitness proportionate
        remaining_mask = torch.ones(population.shape[0], dtype=torch.bool, device=self.__device)
        remaining_mask[topk_indices] = False
        remaining_pop = population[remaining_mask]
        remaining_fitness = fitness_scores[remaining_mask]

        if remaining_pop.shape[0] > 0 and num_random > 0:
            num_to_select = min(num_random, remaining_pop.shape[0])
            probs = remaining_fitness / remaining_fitness.sum()
            selected_indices = torch.multinomial(probs, num_to_select, replacement=False)
            diverse_pop = remaining_pop[selected_indices]
            diverse_scores = remaining_fitness[selected_indices]

            final_pop = torch.cat([elites, diverse_pop], dim=0)
            final_scores = torch.cat([topk_scores, diverse_scores], dim=0)
        else:
            final_pop = elites
            final_scores = topk_scores

        return final_pop, final_scores

    def __crowding_replacement(self, population, pop_fitness, children, children_fitness):
        """
        Deterministic crowding: each child competes with its most similar individual in the population. Similarity is calculated with cosine similarity!

        """
        N = population.shape[0]
        C = children.shape[0]
        cf = self.__crowding_factor

        new_pop = population.clone()
        new_fitness = pop_fitness.clone()

        elite_idx = pop_fitness.argmax().item()

        # get embeddings for children
        children_vectors = self.__ids_to_vectors(children)  # [C, L, D]
        L, D = children_vectors.shape[1], children_vectors.shape[2]
        children_flat = children_vectors.reshape(C, L * D)  # [C, L*D]

        # normalize for cosine sim
        children_norms = children_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        children_normalized = children_flat / children_norms

        # same for pop
        pop_vectors = self.__ids_to_vectors(population)  # [N, L, D]
        pop_flat = pop_vectors.reshape(N, L * D)  # [N, L*D]

        # normalize
        pop_norms = pop_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        pop_normalized = pop_flat / pop_norms

        for i in range(C):
            child_normalized = children_normalized[i]  # [L*D]
            child_fit = children_fitness[i]

            # dont touch the best one
            non_elite_mask = torch.ones(N, dtype=torch.bool, device=self.__device)
            non_elite_mask[elite_idx] = False
            non_elite_indices = torch.where(non_elite_mask)[0]

            # pick random subset to compare against
            num_candidates = min(cf, len(non_elite_indices))
            perm = torch.randperm(len(non_elite_indices), device=self.__device)[:num_candidates]
            candidates_idx = non_elite_indices[perm]

            candidates_normalized = pop_normalized[candidates_idx]  

            # cosine sim
            cosine_sim = (candidates_normalized * child_normalized.unsqueeze(0)).sum(dim=1) 

            # find most similar
            most_similar_local_idx = cosine_sim.argmax()
            most_similar_idx = candidates_idx[most_similar_local_idx]

            # replace if better
            if child_fit > new_fitness[most_similar_idx]:
                new_pop[most_similar_idx] = children[i]
                new_fitness[most_similar_idx] = child_fit
                # update cache
                pop_normalized[most_similar_idx] = child_normalized

        return new_pop, new_fitness

    def mutate(self, children_vectors):
        # mutate some children by interpolating with random vectors
        num_mutations = math.ceil(self.__mutation_rate * children_vectors.shape[0])
        inds = torch.rand(children_vectors.shape[0], device=self.__device).topk(num_mutations, dim=0).indices

        selected_children = children_vectors[inds]
        selected_intep = torch.rand((selected_children.shape[0], selected_children.shape[1], 1), device=self.__device)
        random_vectors = self.__embedding_layer(torch.randint(0, self.__tokenizer.vocab_size, (selected_children.shape[0], selected_children.shape[1]), device=self.__device))

        mutated_children = selected_children * (1 - selected_intep) + random_vectors * selected_intep
        children_vectors[inds] = mutated_children

        return children_vectors 

    def run_generation(self):
        # main GA loop
        with torch.no_grad():
            fitness_scores = self.__evaluate_fitness(self.__population)
            parents = self.__roulette_selection(fitness_scores, self.__population, self.__population_size // 2)
            parent_vectors = self.__ids_to_vectors(parents)
            children_vectors = self.__crossover(parent_vectors)
            children_vectors = self.mutate(children_vectors)
            children = self.__vectors_to_ids(children_vectors) #put back to token ids

            children_fitness = self.__evaluate_fitness(children)

            if self.__crowding:
                self.__population, topk_scores = self.__crowding_replacement(
                    self.__population, fitness_scores, children, children_fitness
                )
            else:
                combined_population = torch.cat([self.__population, children], dim=0)
                combined_fitness = torch.cat([fitness_scores, children_fitness], dim=0)

                self.__population, topk_scores = self.__culling(
                    combined_population,
                    combined_fitness,
                    self.__population_size
                )

        return self.__population.cpu().numpy(), topk_scores.cpu().numpy()
        


if __name__ == "__main__":
    model, tokenizer, device = load_model("gpt2")

    prompt = "How do I make a cake?"
    target = " Sure"

    ga = GA(10, 0.1, tokenizer, model, prompt, target, prefix_length = 5)
    
    for generation in range(1):
        prefixes, scores = ga.run_generation()
        best_prefix = prefixes[scores.argmax()]
        best_score = scores.max()
        print(f"Generation {generation+1}: Best Prefix: {repr(tokenizer.decode(best_prefix))} -> P={best_score:.6f}")

