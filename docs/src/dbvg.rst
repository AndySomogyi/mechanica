.. default-domain:: cpp


Dynamic Bounding Volume Tree
============================

Bullet contains an optimized dynamic bounding volume data structure, :class:`btDbvt`


Stuff about Test2

.. class:: Test2

	   Stuff

	   .. function:: Test2(int a)

      A function that does stuff

   .. member:: Test2 foo

      A member that does more stuff

   
      



.. class:: btDbvt
The btDbvt class implements a fast dynamic bounding volume tree based on axis aligned bounding boxes (aabb tree).
This btDbvt is used for soft body collision detection and for the btDbvtBroadphase. It has a fast insert, remove and update of nodes.
Unlike the btQuantizedBvh, nodes can be dynamically moved around, which allows for change in
topology of the underlying data structure.




.. class:: btDbvt::sStkNN

Stack element

.. member:: const btDbvtNode*	a;

.. var:: const btDbvtNode*	b;
	
.. class:: sStkNP

      .. var:: const btDbvtNode*	node;
      .. var:: int			mask;

      ..


		  struct	sStkNPS
	{
		const btDbvtNode*	node;
		int			mask;
		btScalar	value;
		sStkNPS() {}
		sStkNPS(const btDbvtNode* n,unsigned m,btScalar v) : node(n),mask(m),value(v) {}
	};
	struct	sStkCLN
	{
		const btDbvtNode*	node;
		btDbvtNode*		parent;
		sStkCLN(const btDbvtNode* n,btDbvtNode* p) : node(n),parent(p) {}
	};
	// Policies/Interfaces

	/* ICollide	*/ 
	struct	ICollide
	{		
		DBVT_VIRTUAL_DTOR(ICollide)
			DBVT_VIRTUAL void	Process(const btDbvtNode*,const btDbvtNode*)		{}
		DBVT_VIRTUAL void	Process(const btDbvtNode*)					{}
		DBVT_VIRTUAL void	Process(const btDbvtNode* n,btScalar)			{ Process(n); }
		DBVT_VIRTUAL bool	Descent(const btDbvtNode*)					{ return(true); }
		DBVT_VIRTUAL bool	AllLeaves(const btDbvtNode*)					{ return(true); }
	};
	/* IWriter	*/ 
	struct	IWriter
	{
		virtual ~IWriter() {}
		virtual void		Prepare(const btDbvtNode* root,int numnodes)=0;
		virtual void		WriteNode(const btDbvtNode*,int index,int parent,int child0,int child1)=0;
		virtual void		WriteLeaf(const btDbvtNode*,int index,int parent)=0;
	};
	/* IClone	*/ 
	struct	IClone
	{
		virtual ~IClone()	{}
		virtual void		CloneLeaf(btDbvtNode*) {}
	};

	// Constants
	enum	{
		SIMPLE_STACKSIZE	=	64,
		DOUBLE_STACKSIZE	=	SIMPLE_STACKSIZE*2
	};

	// Fields
	btDbvtNode*		m_root;
	btDbvtNode*		m_free;
	int				m_lkhd;
	int				m_leaves;
	unsigned		m_opath;

	
	btAlignedObjectArray<sStkNN>	m_stkStack;


	// Methods

	void			clear();
	bool			empty() const;
	void			optimizeBottomUp();
	void			optimizeTopDown(int bu_treshold=128);
	void			optimizeIncremental(int passes);
	btDbvtNode*		insert(const btDbvtVolume& box,void* data);
	void			update(btDbvtNode* leaf,int lookahead=-1);
	void			update(btDbvtNode* leaf,btDbvtVolume& volume);
	bool			update(btDbvtNode* leaf,btDbvtVolume& volume,const btVector3& velocity,btScalar margin);
	bool			update(btDbvtNode* leaf,btDbvtVolume& volume,const btVector3& velocity);
	bool			update(btDbvtNode* leaf,btDbvtVolume& volume,btScalar margin);	
	void			remove(btDbvtNode* leaf);
	void			write(IWriter* iwriter) const;
	void			clone(btDbvt& dest,IClone* iclone=0) const;
	static int		maxdepth(const btDbvtNode* node);
	static int		countLeaves(const btDbvtNode* node);
	static void		extractLeaves(const btDbvtNode* node,btAlignedObjectArray<const btDbvtNode*>& leaves);

	static void		benchmark();


		static void		enumNodes(	const btDbvtNode* root,
		ICollide& policy);
	
		static void		enumLeaves(	const btDbvtNode* root,
		ICollide& policy);
	
		void		collideTT(	const btDbvtNode* root0,
		const btDbvtNode* root1,
		ICollide& policy);

	
		void		collideTTpersistentStack(	const btDbvtNode* root0,
		  const btDbvtNode* root1,
		  ICollide& policy);

	
		void		collideTV(	const btDbvtNode* root,
		const btDbvtVolume& volume,
		ICollide& policy) const;
	
	
	 void		collideTVNoStackAlloc(	const btDbvtNode* root,
						  const btDbvtVolume& volume,
						  btNodeStack& stack,
						  ICollide& policy) const;
	
	
	
	
	 ///rayTest is a re-entrant ray test, and can be called in parallel as long as the btAlignedAlloc is thread-safe (uses locking etc)
	 ///rayTest is slower than rayTestInternal, because it builds a local stack, using memory
	 allocations, and it recomputes signs/rayDirectionInverses each time
	
	
		static void		rayTest(	const btDbvtNode* root,
		const btVector3& rayFrom,
		const btVector3& rayTo,
		ICollide& policy);
	///rayTestInternal is faster than rayTest, because it uses a persistent stack (to reduce dynamic memory allocations to a minimum) and it uses precomputed signs/rayInverseDirections
	///rayTestInternal is used by btDbvtBroadphase to accelerate world ray casts
	
	
		void		rayTestInternal(	const btDbvtNode* root,
								const btVector3& rayFrom,
								const btVector3& rayTo,
								const btVector3& rayDirectionInverse,
								unsigned int signs[3],
								btScalar lambda_max,
								const btVector3& aabbMin,
								const btVector3& aabbMax,
                                btAlignedObjectArray<const btDbvtNode*>& stack,
								ICollide& policy) const;

	
		static void		collideKDOP(const btDbvtNode* root,
		const btVector3* normals,
		const btScalar* offsets,
		int count,
		ICollide& policy);
	
		static void		collideOCL(	const btDbvtNode* root,
		const btVector3* normals,
		const btScalar* offsets,
		const btVector3& sortaxis,
		int count,								
		ICollide& policy,
		bool fullsort=true);
	
		static void		collideTU(	const btDbvtNode* root,
		ICollide& policy);


	static  int	nearest(const int* i,const btDbvt::sStkNPS* a,btScalar v,int l,int h)

	static  int	allocate(	btAlignedObjectArray<int>& ifree,
		btAlignedObjectArray<sStkNPS>& stock,
		const sStkNPS& value)


