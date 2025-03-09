# Synthesizing λ-join-calculus behaviors in Z3

This project aims to synthesize λ-join-calculus behaviors using the Z3 SMT solver.

The λ-calculus is a simple programming language.
The λ-join-calculus extends the λ-calculus with a join operator with respect to the Scott ordering.
Let's define a _behavior_ as the equivalence class of _programs_ modulo the coarsest sensible equivalence relation on programs, namely Hyland and Wadsworth's H* theory of
observational equivalence, corresponding to Scott's D∞ models:
```
M ⊑ N iff ∀ context C[ ], C[M] converges ⇒ C[N] converges
M ≡ N iff ∀ context C[ ], C[M] converges ⇔ C[N] converges 
```

We'll leverage Scott's types-as-closures framework that naturally arises when
extending the λ-calculus with join.
Closures will allow us to branch on finitely or countably many inhabitants of a type.

Our synthesis approach is:
- Input a program sketch and a set of constraints.
- Explore a space of linear normal forms refining the sketch.
- Narrow down the space of programs by checking constraints, using Z3 to weakly
  check satisfiability.
- Within the feasible search space, use CEGIS (Counterexample-Guided Inductive
  Synthesis) to synthesize a chain of increasingly feasible programs.

## Examples

Our first challenge problem is to synthesize a finitary definition for the
simple type constructor
```
SIMPLE = ⨆ { <r,s> | s ◦ r ⊑ I }
```
where `⊑` is the Scott ordering, `⨆` is the infinitary join operator, `I` is the
identity function, `◦` is function composition, `<r,s> = (λx. x r s)` is a pair,
and `r` and `s` range over closed λ-join-calculus terms.

## Performance engineering resources

- F* authors (20??) _Profiling Z3 and Solving Proof Performance Issues_ ([html](https://fstar-lang.org/tutorial/book/under_the_hood/uth_smt.html#profiling-z3-and-solving-proof-performance-issues)|[rst](https://github.com/FStarLang/PoP-in-FStar/tree/main/book/under_the_hood))
- Michał Moskal (2009) _Programming with Triggers_ ([pdf](https://mmoskal.github.io/pdf/prtrig.pdf))
- Oskari Jyrkinen, Jonáš Fiala, et al. (2023) _SMT Scope_ ([code](https://github.com/viperproject/smt-scope)|[webapp](https://viperproject.github.io/smt-scope/))
- Nils Becker, Peter Müller, and Alexander J. Summers (2019) _The Axiom Profiler: Understanding and Debugging SMT Quantifier Instantiations_ ([pdf](https://pm.inf.ethz.ch/publications/BeckerMuellerSummers19.pdf))
- Nikolaj Bjørner, Leonardo de Moura, Lev Nachmanson, and Christoph Wintersteiger (20??) _Programming Z3_ ([html](https://z3prover.github.io/papers/programmingz3.html))
- Nikolaj Bjørner et. al (20??) _Z3 Internals (Draft)_ ([html](https://z3prover.github.io/papers/z3internals.html))
- Nikolaj Bjørner et. al (2021) _Supercharging Plant Configurations using Z3_ ([html](https://z3prover.github.io/papers/Supercharging.html))
- Leonardo de Moura, Nikolaj Bjørner (2008) _Z3: An Efficient SMT Solver_ ([pdf](https://link.springer.com/content/pdf/10.1007/978-3-540-78800-3_24.pdf))
- Z3 github issues ([link](https://github.com/Z3Prover/z3/issues?q=performance))