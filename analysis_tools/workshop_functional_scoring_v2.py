"""Symmetric scoring corrections; queries/cohort/attack selection stay frozen.

v1 is retained verbatim and every original saved score is validated against it.
Corrections address false negatives found in an output spot check, before final
recovery evaluation: schema 'does not have extra properties' is not an invalid
verdict; an unbracketed preference can be task-correct but contract-incorrect;
code assertions may be fenced with a prose preface and integer expressions.
Both versions are reported, and no outputs are regenerated to implement this.
"""
import ast
import re
from analysis_tools.workshop_functional_benchmark import score as score_v1,unfence

CHANGES={
 'schema':'Replace blanket does-not negation by schema-verdict-specific negation.',
 'preference':'Separate recoverable A/B choice correctness from exact [[A]]/[[B]] contract.',
 'assertions':'Parse fenced Python within prose and bounded literal integer arithmetic; never execute model code.'
}

def integer(node,depth=0):
    if depth>12:raise ValueError('too deep')
    if isinstance(node,ast.Constant) and type(node.value) is int:v=node.value
    elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        v=integer(node.operand,depth+1)*(1 if isinstance(node.op,ast.UAdd) else -1)
    elif isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Pow)):
        a=integer(node.left,depth+1);b=integer(node.right,depth+1)
        if isinstance(node.op,ast.Add):v=a+b
        elif isinstance(node.op,ast.Sub):v=a-b
        elif isinstance(node.op,ast.Mult):v=a*b
        else:
            if not 0<=b<=63 or abs(a)>2**32:raise ValueError('unsafe exponent')
            v=a**b
    else:raise ValueError('not a literal integer expression')
    if abs(v)>2**64:raise ValueError('integer bound')
    return v

def score(task,row,text):
    if task not in CHANGES:return score_v1(task,row,text)
    old=score_v1(task,row,text);s=unfence(text);low=s.lower();g=row['gold'];t=False;c=bool(old['contract'])
    if task=='schema':
        positive=bool(re.search(r'\bvalid\b|\bsatisfies\b|\bconforms\b|\bcomplies\b|\bpasses\b',low))
        negative=bool(re.search(r'\binvalid\b|\bnot\s+valid\b|\b(?:does not|doesn.t|cannot|fails? to)\s+(?:satisfy|conform|comply|meet|match)\b|\b(?:violates?|violated|violating)\s+(?:the\s+)?(?:schema|constraint|requirement)\b|\b(?:fails?|failed)\s+(?:the\s+)?(?:schema\s+)?validation\b',low))
        if g['valid']:t=positive and not negative
        else:t=negative and g['property'] in s and 'integer' in low
    elif task=='preference':
        clean=re.sub(r'\*\*([^*]+)\*\*',r'\1',s)
        m=re.match(r'\s*(?:(?:I (?:choose|prefer)|Choice:|Answer:)\s*)?(?:Assistant\s*)?(?:\[\[)?([AB])(?:\]\])?(?:[.!:]|\s|$)',clean,re.I)
        ambiguous='[[A]]' in clean and '[[B]]' in clean
        t=bool(m and m[1].upper()==g['label'] and not ambiguous)
    elif task=='assertions':
        blocks=re.findall(r'```(?:python|Python|py)?\s*\n(.*?)```',text,re.S)
        code='\n'.join(blocks) if blocks else s
        try:tree=ast.parse(code)
        except SyntaxError:tree=ast.Module(body=[],type_ignores=[])
        assertions=[n for n in ast.walk(tree) if isinstance(n,ast.Assert)];values=[];valid=0
        for node in assertions:
            x=node.test
            try:
                assert isinstance(x,ast.Compare) and len(x.ops)==1 and isinstance(x.ops[0],ast.Eq)
                call=x.left;assert isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id==g['function'] and len(call.args)==1 and not call.keywords
                a=integer(call.args[0]);b=integer(x.comparators[0]);values.append(a);valid+=int(b==a+g['offset'])
            except (AssertionError,ValueError,TypeError):pass
        c=len(assertions)==10 and len(set(values))==10;t=c and valid==10
    return {'task':float(t),'contract':float(c),'joint':float(t and c)}
