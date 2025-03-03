# Synthesizing λ-join-calculus behaviors in Z3

This project aims to synthesize λ-join-calculus behaviors using the Z3 SMT solver.

The λ-calculus is a simple programming language.
The λ-join-calculus extends the λ-calculus with a join operator WRT the Scott ordering.
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

Our first challenge problem is to synthesize a finitary definition for the
simple type constructor
```
SIMPLE = ⨆ { <r,s> | s ◦ r ⊑ I }
```
where `⊑` is the Scott ordering, `⨆` is the infinitary join operator, `I` is the
identity function, `◦` is function composition, `<r,s> = (λx. x r s)` is a pair,
and `r` and `s` range over closed λ-join-calculus terms.
