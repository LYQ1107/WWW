import torch
from torch import nn
from .schemas import validate, OptionTensors,OnlineVisualState
from .visual_state_encoder import StateEncoder,ReaderBlock
from .typed_question_reader import TypedQuestionReader
from .action_option_encoder import ActionOptionEncoder
from .option_reader import ActionOptionReader
from .typed_decision_heads import TypedDecisionHeads
from .consequence_value_head import ConsequenceValueHead

class VisualJev(nn.Module):
    def __init__(self,variant='full',d=128,heads=4):
        super().__init__()
        supported={'full','set_transformer','visual_deepsets','question_plain','no_question_reader','fixed_question','no_option_reader','no_gating','no_shared_state','no_history','no_H32','no_consequence','similarity'}
        if variant not in supported:
            raise ValueError('unknown preregistered variant')
        self.variant=variant
        self.state_encoder=StateEncoder(d,heads)
        self.question_reader=TypedQuestionReader(d,heads)
        self.option_encoder=ActionOptionEncoder(d)
        self.option_reader=ActionOptionReader(d,heads) if variant not in {'set_transformer','visual_deepsets','question_plain'} else None
        if variant=='question_plain':
            self.plain_head=nn.Sequential(nn.Linear(2*d,4*d),nn.GELU(),nn.Linear(4*d,4*d),nn.GELU(),nn.Linear(4*d,d),nn.LayerNorm(d))
        if variant=='set_transformer':
            self.candidate_blocks=nn.ModuleList([ReaderBlock(d,heads),ReaderBlock(d,heads)])
        if variant=='visual_deepsets':
            self.candidate_blocks=nn.Sequential(nn.Linear(3*d,4*d),nn.GELU(),nn.Linear(4*d,d),nn.GELU(),nn.Linear(d,d))
        self.heads=TypedDecisionHeads(d)
        self.consequence=ConsequenceValueHead(d)
        self.similarity_projection=nn.Linear(d,d,bias=False) if variant=='similarity' else None
        for name,width in [('state',64),('evidence',12)]:
            self.register_buffer(name+'_mean',torch.zeros(width))
            self.register_buffer(name+'_std',torch.ones(width))
            self.register_buffer(name+'_constant',torch.zeros(width,dtype=torch.bool))

    def normalized(self,x,name):
        result=(x-getattr(self,name+'_mean'))/getattr(self,name+'_std').clamp_min(1e-8)
        return result.masked_fill(getattr(self,name+'_constant'),0.)

    def encode_state(self,state):
        memory,mask=self.state_encoder(state)
        if self.variant=='no_shared_state':
            memory=memory*0
        return memory,mask

    def encode_questions(self,state_memory,questions):
        mode='fixed' if self.variant=='fixed_question' else 'no_reader' if self.variant in {'no_question_reader','set_transformer','visual_deepsets'} else 'dynamic'
        return self.question_reader(*state_memory,questions,self.normalized(questions.context,'state'),mode=mode)

    def score_questions(self,state_memory,question,options,types):
        memory,mask=state_memory
        if self.variant=='no_history':
            # Only actual current query observation remains, injected by caller
            # in the ablation input builder; zero history here is a second guard.
            visual=options.visual*0
            options=OptionTensors(visual,options.visual_mask,options.evidence,options.kinds,options.mask)
        encoded=self.option_encoder(question,options,self.normalized(options.evidence,'evidence'))
        if self.variant=='set_transformer':
            b,q,k,d=encoded.shape
            actions=encoded+question[:,:,None,:]
            if k and q:
                x=actions.reshape(b*q,k,d);legal=options.mask.reshape(b*q,k)
                safe=legal.clone();safe[~safe.any(-1),0]=True
                for block in self.candidate_blocks:x=block(x,x,safe)
                actions=x.reshape(b,q,k,d)*options.mask[...,None]
        elif self.variant=='visual_deepsets':
            count=options.mask.sum(-1,keepdim=True).clamp_min(1)
            mean=(encoded*options.mask[...,None]).sum(-2)/count
            context=mean[:,:,None,:].expand_as(encoded)
            actions=self.candidate_blocks(torch.cat((encoded,context,question[:,:,None,:].expand_as(encoded)),-1))
        elif self.variant=='question_plain':
            actions=self.plain_head(torch.cat((encoded,question[:,:,None,:].expand_as(encoded)),-1))
        else:
            actions=self.option_reader(question,memory,mask,encoded,options.mask,gating=self.variant!='no_gating',read=self.variant not in {'no_option_reader','question_plain'})
        out=self.heads(actions,question,types,options.kinds,options.mask)
        if self.variant=='similarity':
            from .schemas import masked_softmax
            logits=(self.similarity_projection(actions)*question[:,:,None,:]).sum(-1)/(question.shape[-1]**.5)
            out['choice_logits']=logits
            out['probabilities']=masked_softmax(logits,options.mask)
        out.update(question_state=question,action_representations=actions,consequences=self.consequence(actions),state_memory=memory)
        return out

    def forward(self,state,questions,options):
        validate(state,questions,options)
        if self.variant=='no_history':
            current=state.mask&(state.metadata[...,0]==1)
            empty=~current.any(-1)
            current=current.clone();current[empty,0]=True
            visual=state.visual*(state.metadata[...,0]==1)[...,None]
            state=OnlineVisualState(visual,state.metadata,current)
        memory=self.encode_state(state)
        question=self.encode_questions(memory,questions)
        return self.score_questions(memory,question,options,questions.types)
