import re
from sge.utilities import ordered_set
import json
import numpy as np
class Grammar:
    """Class that represents a grammar. It works with the prefix notation."""
    NT = "NT"
    T = "T"
    NT_PATTERN = "(<.+?>)"
    RULE_SEPARATOR = "::="
    PRODUCTION_SEPARATOR = "|"

    def __init__(self):
        self.grammar_file = None
        self.grammar = {}
        self.productions_labels = {}
        self.non_terminals, self.terminals = set(), set()
        self.ordered_non_terminals = ordered_set.OrderedSet()
        self.non_recursive_options = {}
        self.number_of_options_by_non_terminal = None
        self.start_rule = None
        self.max_depth = None
        self.max_init_depth = None
        self.max_number_prod_rules = 0
        self.pcfg = None
        self.pcfg_mask = None
        self.pcfg_path = None
        self.index_of_non_terminal = {}
        self.shortest_path = {}

    def set_path(self, grammar_path):
        self.grammar_file = grammar_path

    def set_pcfg_path(self, pcfg_path):
        self.pcfg_path = pcfg_path

    def set_min_init_tree_depth(self, min_tree_depth):
        self.max_init_depth = min_tree_depth

    def set_max_tree_depth(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def get_max_depth(self):
        return self.max_depth
    
    def get_max_init_depth(self):
        return self.max_init_depth

    def read_grammar(self):
        """
        Reads a Grammar in the BNF format and converts it to a python dictionary
        This method was adapted from PonyGE version 0.1.3 by Erik Hemberg and James McDermott
        """
        if self.grammar_file is None:
            raise Exception("You need to specify the path of the grammar file")


        with open(self.grammar_file, "r") as f:
            for line in f:
                if not line.startswith("#") and line.strip() != "":
                    if line.find(self.PRODUCTION_SEPARATOR):
                        left_side, productions = line.split(self.RULE_SEPARATOR)
                        left_side = left_side.strip()
                        if not re.search(self.NT_PATTERN, left_side):
                            raise ValueError("Left side not a non-terminal!")
                        self.non_terminals.add(left_side)
                        self.ordered_non_terminals.add(left_side)
                        # assumes that the first rule in the file is the axiom
                        if self.start_rule is None:
                            self.start_rule = (left_side, self.NT)
                        temp_productions = []
                        for production in [production.strip() for production in productions.split(self.PRODUCTION_SEPARATOR)]:
                            temp_production = []
                            if not re.search(self.NT_PATTERN, production):
                                if production == "None":
                                    production = ""
                                self.terminals.add(production)
                                temp_production.append((production, self.T))
                            else:
                                for value in re.findall("<.+?>|[^<>]*", production):
                                    if value != "":
                                        if re.search(self.NT_PATTERN, value) is None:
                                            sym = (value, self.T)
                                            self.terminals.add(value)
                                        else:
                                            sym = (value, self.NT)
                                        temp_production.append(sym)
                            temp_productions.append(temp_production)                          
                        self.max_number_prod_rules = max(self.max_number_prod_rules, len(temp_productions))
                        if left_side not in self.grammar:
                            self.grammar[left_side] = temp_productions
        
        self.generate_uniform_pcfg()
        if self.pcfg_path is not None:
            # load PCFG probabilities from json file. List of lists, n*n, with n = max number of production rules of a NT
            with open(self.pcfg_path) as f:
                self.pcfg = np.array(json.load(f))
        # self.compute_non_recursive_options()
        self.find_shortest_path()


    def find_shortest_path(self):
        open_symbols = []
        for nt in self.grammar.keys():
            depth = self.minimum_path_calc((nt,'NT'), open_symbols)
            
    def minimum_path_calc(self, current_symbol, open_symbols):
        if current_symbol[1] == self.T:
            return 0
        else:
            open_symbols.append(current_symbol)
            for derivation_option in self.grammar[current_symbol[0]]:
                max_depth = 0
                if current_symbol not in self.shortest_path:
                    self.shortest_path[current_symbol] = [999999]
                if bool(sum([i in open_symbols for i in derivation_option])):
                    continue
                if current_symbol not in derivation_option:
                    for symbol in derivation_option:
                        depth = self.minimum_path_calc(symbol, open_symbols)
                        depth += 1
                        if depth > max_depth:
                            max_depth = depth

                    if max_depth < self.shortest_path[current_symbol][0]:
                        self.shortest_path[current_symbol] = [max_depth]
                        if derivation_option not in self.shortest_path[current_symbol]:
                            self.shortest_path[current_symbol].append(derivation_option)
                    if max_depth == self.shortest_path[current_symbol][0]:
                        if derivation_option not in self.shortest_path[current_symbol]:
                            self.shortest_path[current_symbol].append(derivation_option)
            open_symbols.remove(current_symbol)
            return self.shortest_path[current_symbol][0]
                    
            

    def create_counter(self):
        self.counter = dict.fromkeys(self.grammar.keys(),[])
        for k in self.counter.keys():
            self.counter[k] = [0] * len(self.grammar[k])

    def generate_uniform_pcfg(self):
        """
        assigns uniform probabilities to grammar
        """
        array = np.zeros(shape=(len(self.grammar.keys()),self.max_number_prod_rules))
        for i, nt in enumerate(self.grammar):
            number_probs = len(self.grammar[nt])
            prob = 1.0 / number_probs
            array[i,:number_probs] = prob
            if nt not in self.index_of_non_terminal:
                self.index_of_non_terminal[nt] = i
        self.pcfg = array
        self.pcfg_mask = self.pcfg != 0

    def generate_random_pcfg(self):
        pass

    def get_mask(self):
        return self.pcfg_mask

    def get_index_of_non_terminal(self):
        return self.index_of_non_terminal

    def get_non_terminals(self):
        return self.ordered_non_terminals

    def count_number_of_options_in_production(self):
        if self.number_of_options_by_non_terminal is None:
            self.number_of_options_by_non_terminal = {}
            for nt in self.ordered_non_terminals:
                self.number_of_options_by_non_terminal.setdefault(nt, len(self.grammar[nt]))
        return self.number_of_options_by_non_terminal

    def list_non_recursive_productions(self, nt):
        non_recursive_elements = []
        for options in self.grammar[nt]:
            for option in options:
                if option[1] == self.NT and option[0] == nt:
                    break
            else:
                non_recursive_elements += [options]
        return non_recursive_elements
    
    def get_probability(self, grammar, nt_index, index):
        return grammar[nt_index,index]
        
    def get_probabilities_non_terminal(self, grammar, nt_index):
        return grammar[nt_index]

    def recursive_individual_creation(self, genome, symbol, current_depth):
        codon = np.random.uniform()
        nt_index = self.index_of_non_terminal[symbol]
        if current_depth > self.max_init_depth:
            shortest_path = self.shortest_path[(symbol,'NT')]
            prob_non_recursive = 0.0
            for rule in shortest_path[1:]:
                index = self.grammar[symbol].index(rule)
                prob_non_recursive += self.pcfg[nt_index,index]
            prob_aux = 0.0
            for rule in shortest_path[1:]:
                index = self.grammar[symbol].index(rule)
                new_prob = self.pcfg[nt_index,index] / prob_non_recursive
                prob_aux += new_prob
                if codon <= round(prob_aux,3):
                    expansion_possibility = index
                    break
        else:
            prob_aux = 0.0
            for index in range(len(self.grammar[symbol])):
                prob_aux += self.pcfg[nt_index,index]
                if codon <= round(prob_aux,3):
                    expansion_possibility = index
                    break

        genome[self.get_non_terminals().index(symbol)].append([expansion_possibility,codon,current_depth])
        expansion_symbols = self.grammar[symbol][expansion_possibility]
        depths = [current_depth]
        for sym in expansion_symbols:
            if sym[1] != self.T:
                depths.append(self.recursive_individual_creation(genome, sym[0], current_depth + 1))
        return max(depths)

    def mapping(self, mapping_rules, positions_to_map=None, needs_python_filter=False):
        if positions_to_map is None:
            positions_to_map = [0] * len(self.ordered_non_terminals)
        output = []
        max_depth = self._recursive_mapping(mapping_rules, positions_to_map, self.start_rule, 0, output)
        output = "".join(output)
        if self.grammar_file.endswith("pybnf"):
            output = self.python_filter(output, needs_python_filter)
        return output, max_depth

    def _recursive_mapping(self, mapping_rules, positions_to_map, current_sym, current_depth, output):
        depths = [current_depth]
        if current_sym[1] == self.T:
            output.append(current_sym[0])
        else:
            current_sym_pos = self.ordered_non_terminals.index(current_sym[0])
            choices = self.grammar[current_sym[0]]
            shortest_path = self.shortest_path[current_sym]
            nt_index = self.index_of_non_terminal[current_sym[0]]
            if positions_to_map[current_sym_pos] >= len(mapping_rules[current_sym_pos]):
                codon = np.random.uniform()
                if current_depth > self.max_depth:
                    prob_non_recursive = 0.0
                    for rule in shortest_path[1:]:
                        index = self.grammar[current_sym[0]].index(rule)
                        prob_non_recursive += self.pcfg[nt_index,index]
                    prob_aux = 0.0
                    for rule in shortest_path[1:]:
                        index = self.grammar[current_sym[0]].index(rule)
                        new_prob = self.pcfg[nt_index,index] / prob_non_recursive
                        prob_aux += new_prob
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                else:
                    prob_aux = 0.0
                    for index, option in enumerate(self.grammar[current_sym[0]]):
                        prob_aux += self.pcfg[nt_index,index]
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                mapping_rules[current_sym_pos].append([expansion_possibility,codon,current_depth])
            else:
                # re-mapping with new probabilities                
                codon = mapping_rules[current_sym_pos][positions_to_map[current_sym_pos]][1]
                if current_depth > self.max_depth:
                    prob_non_recursive = 0.0
                    for rule in shortest_path[1:]:
                        index = self.grammar[current_sym[0]].index(rule)
                        prob_non_recursive += self.pcfg[nt_index,index]
                    prob_aux = 0.0
                    for rule in shortest_path[1:]:
                        index = self.grammar[current_sym[0]].index(rule)
                        new_prob = self.pcfg[nt_index,index] / prob_non_recursive
                        prob_aux += new_prob
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
                else:
                    prob_aux = 0.0
                    for index in range(len(self.grammar[current_sym[0]])):
                        prob_aux += self.pcfg[nt_index,index]
                        if codon <= round(prob_aux,3):
                            expansion_possibility = index
                            break
            # update mapping rules com a updated expansion possibility
            mapping_rules[current_sym_pos][positions_to_map[current_sym_pos]] = [expansion_possibility,codon,current_depth]
            current_production = expansion_possibility
            positions_to_map[current_sym_pos] += 1
            next_to_expand = choices[current_production]
            for next_sym in next_to_expand:
                depths.append(
                    self._recursive_mapping(mapping_rules, positions_to_map, next_sym, current_depth + 1, output))
        return max(depths)

    def compute_non_recursive_options(self):
        for key in self.grammar.keys():
            prob_non_recursive = 0.0
            non_recursive_prods = []
            for index, option in enumerate(self.grammar[key]):
                for s in option:
                    if s[0] == key:
                        break
                else:
                    prob_non_recursive += self.pcfg[self.index_of_non_terminal[key],index]
                    non_recursive_prods.append([index, option])
            self.non_recursive_options[key] = [non_recursive_prods, prob_non_recursive]

    def get_non_recursive_options(self, symbol):
        return self.non_recursive_options[symbol]


    def get_dict(self):
        return self.grammar

    def get_pcfg(self):
        return self.pcfg

    def get_shortest_path(self):
        return self.shortest_path

    @staticmethod
    def python_filter(txt, needs_python_filter):
        """ Create correct python syntax.
        We use {: and :} as special open and close brackets, because
        it's not possible to specify indentation correctly in a BNF
        grammar without this type of scheme."""
        txt = txt.replace("\le", "<=")
        txt = txt.replace("\ge", ">=")
        txt = txt.replace("\l", "<")
        txt = txt.replace("\g", ">")
        txt = txt.replace("\eb", "|")
        if needs_python_filter:
            indent_level = 0
            tmp = txt[:]
            i = 0
            while i < len(tmp):
                tok = tmp[i:i+2]
                if tok == "{:":
                    indent_level += 1
                elif tok == ":}":
                    indent_level -= 1
                tabstr = "\n" + "  " * indent_level
                if tok == "{:" or tok == ":}" or tok == "\\n":
                    tmp = tmp.replace(tok, tabstr, 1)
                i += 1
                # Strip superfluous blank lines.
                txt = "\n".join([line for line in tmp.split("\n") if line.strip() != ""])
        return txt

    def get_start_rule(self):
        return self.start_rule

    def __str__(self):
        grammar = self.grammar
        text = ""
        for key in self.ordered_non_terminals:
            text += key + " ::= "
            for options in grammar[key]:
                for option in options:
                    text += option[0]
                if options != grammar[key][-1]:
                    text += " | "
            text += "\n"
        return text

# Create one instance and export its methods as module-level functions.
# The functions share state across all uses
# (both in the user's code and in the Python libraries), but that's fine
# for most programs and is easier for the casual user


_inst = Grammar()
set_path = _inst.set_path
set_pcfg_path = _inst.set_pcfg_path
read_grammar = _inst.read_grammar
get_non_terminals = _inst.get_non_terminals
count_number_of_options_in_production = _inst.count_number_of_options_in_production
list_non_recursive_productions = _inst.list_non_recursive_productions
recursive_individual_creation = _inst.recursive_individual_creation
mapping = _inst.mapping
start_rule = _inst.get_start_rule
set_max_tree_depth = _inst.set_max_tree_depth
set_min_init_tree_depth = _inst.set_min_init_tree_depth
get_max_depth = _inst.get_max_depth
get_non_recursive_options = _inst.get_non_recursive_options
# compute_non_recursive_options = _inst.compute_non_recursive_options
get_dict = _inst.get_dict
get_pcfg = _inst.get_pcfg
get_probability = _inst.get_probability
get_probabilities_non_terminal = _inst.get_probabilities_non_terminal
get_mask = _inst.get_mask
get_shortest_path = _inst.get_shortest_path
get_index_of_non_terminal = _inst.get_index_of_non_terminal
ordered_non_terminals = _inst.ordered_non_terminals
max_init_depth = _inst.get_max_init_depth
python_filter = _inst.python_filter
if __name__ == "__main__":
    np.random.seed(42)
    g = Grammar("grammars/regression.txt", 9)
    genome = [[0], [0, 3, 3], [0], [], [1, 1]]
    mapping_numbers = [0] * len(genome)
    print(g.mapping(genome, mapping_numbers, needs_python_filter=True))

